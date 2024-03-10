import re
from datetime import datetime, timedelta
from enum import Enum
from numbers import Number
from typing import Optional

import appdaemon.plugins.hass.hassapi as hass
import numpy as np
import pandas as pd
import pytz
from sqlalchemy import create_engine


class Automations(hass.Hass):
    """Automations class."""

    def initialize(self):
        """Initialize the HACS app."""
        self.log("Smart Home Automation - ReGex Entity Matching and Command Execution")
        self.connect_to_database()
        self.get_hass_data()
        self.create_home_automation_settings()
        self.recordings = {}
        self.debounce_timers = {}
        self.debounce_period = timedelta(seconds=60)
        # self.begin_snapshot(
        #     entity_id='binary_sensor.office_occupancy_desk_gerardo',
        #     old='off',
        #     new='on',
        #     recording_key='gerardo_coding',
        #     regex_filter=None,
        #     duration=10,
        # )

    def begin_snapshot(self, entity_id, old, new, recording_key, regex_filter, duration):
        self.listen_state(
            self.trigger_event_recorder,
            entity_id,
            old=old,
            new=new,
            recording_key=recording_key,
            regex_filter=regex_filter,
            duration=duration,
        )

    def should_debounce(self, debounce_key):
        """Check if the call should be debounced."""
        current_time = datetime.now()
        last_run = self.debounce_timers.get(debounce_key)

        if last_run is None:
            # If it's the first run, do not debounce
            self.debounce_timers[debounce_key] = current_time
            return False

        if current_time - last_run < self.debounce_period:
            # If we're within the debounce period, skip this call
            return True

        # If we're outside the debounce period, proceed with the call
        self.debounce_timers[debounce_key] = current_time
        return False

    def connect_to_database(self):
        self.hass_engine = create_engine(self.args['hass_db_url'])
        self.home_engine = create_engine(self.args['home_db_url'])

    def get_data(self, sql, db='hass'):
        try:
            if db == 'hass':
                with self.hass_engine.connect() as connection:
                    return pd.read_sql(sql=sql, con=connection).fillna(0)

            elif db == 'home':
                with self.home_engine.connect() as connection:
                    return pd.read_sql(sql=sql, con=connection).fillna(0)

        except Exception as e:
            self.log(f"Error in running query: {e}", level="ERROR")
            return pd.DataFrame()

    def get_hass_data(self):
        self.areas = {area: area.replace('_', ' ').title() for area in self.args['areas']}

    def create_home_automation_settings(self):
        # Input Selects
        input_selects = {
            'house_mode': [
                'Day',
                'Night',
                'Eco'
            ],

        }
        # Input Booleans
        input_booleans = [
            'entertainment_mode',
            'quiet_mode',
            'privacy_mode',
            'color_notification_is_running',
            'tts_notification_is_running',
        ]

        for select, options in input_selects.items():
            if not self.get_state(f'input_select.{select}'):
                self.set_state(
                    f'input_select.{select}',
                    options=options,
                    state='Day',
                    friendly_name=select.replace('_', ' ').title()
                )

        for boolean in input_booleans:
            self.set_state(f'input_boolean.{boolean}', state='off', friendly_name=boolean.replace('_', ' ').title())

    def _get_states(self, domain=None):
        all_entities = self.get_state()
        if domain:
            # Ensure domain is a list to simplify processing
            if not isinstance(domain, list):
                domain = [domain]
            # Filter entities that start with any of the domains in the list
            all_entities = {entity: value for entity, value in all_entities.items() if
                            any(entity.startswith(f'{d}.') for d in domain)}

        return all_entities

    def _build_matching_entities(self, pattern, pattern_overwrite, area, domain, include_manual_entities,
                                 exclude_manual_entities,
                                 exclude_entities, include_only):
        """Get all entities that match the pattern, area, and domain."""
        entities = self._get_states(domain=domain)
        patterns = []

        if isinstance(pattern, list) or isinstance(area, list) or isinstance(domain, list):
            pattern = pattern if isinstance(pattern, list) else [pattern]
            area = area if isinstance(area, list) else [area]
            domain = domain if isinstance(domain, list) else [domain]

            for p in pattern:
                for a in area:
                    for d in domain:
                        patterns.append(self.generate_patterns(p, a, d))
        else:
            patterns.append(self.generate_patterns(pattern, area, domain))

        if pattern_overwrite is not None:
            patterns = pattern_overwrite if isinstance(pattern_overwrite, list) else [pattern_overwrite]

        matched_entities = []
        if include_manual_entities:
            for entity in include_manual_entities:
                matched_entities.append(entity)
                if entity in exclude_entities:  # Remove entity from exclude if it is in include_manual_entities
                    exclude_entities.pop(entity)

        if not include_only:
            for pattern in patterns:
                for entity in entities:
                    if re.search(pattern, entity) and entity not in matched_entities:
                        matched_entities.append(entity)

        return matched_entities

    def _get_calendar_events(self, entity_id, start_date_time=None, end_date_time=None, duration=None, filter_by=None):
        try:
            if end_date_time is not None:
                calendar_events = self.call_service(
                    domain='calendar',
                    service='get_events',
                    entity_id=entity_id,
                    start_date_time=start_date_time,
                    end_date_time=end_date_time,
                )

            elif duration is not None:
                calendar_events = self.call_service(
                    domain='calendar',
                    service='get_events',
                    entity_id=entity_id,
                    start_date_time=start_date_time,
                    duration=duration,
                )

            else:
                calendar_events = None

            if calendar_events:
                # Assuming `calendar_events` is a dictionary that contains the events under entity's 'events' key
                events = calendar_events.get('events', [])
                approved_operations = {
                    '==': '==' in filter_by,
                    '!=': '!=' in filter_by,
                    '<': '<' in filter_by,
                    '>': '>' in filter_by,
                    'contains': 'contains' in filter_by,
                }

                if filter_by and any(approved_operations.values()):
                    # Convert events to DataFrame for filtering
                    events_df = pd.DataFrame(events)
                    if not events_df.empty:
                        # Convert start and end times to datetime objects for comparison
                        events_df['start'] = pd.to_datetime(events_df['start'])
                        events_df['end'] = pd.to_datetime(events_df['end'])
                        # Apply filter query
                        events_df = events_df.query(filter_by)
                        events = events_df.to_dict(orient='records')
                return events
            else:
                self.log(f"No events found for calendar {entity_id} or error in fetching events.", level="ERROR")
                return None
        except Exception as e:
            self.log(f"Error fetching calendar events for {entity_id}: {str(e)}", level="ERROR")
            return None

    def _get_todo_items(self, entity_id, filter_by=None):
        try:
            todo_items = self.call_service(
                'todo',
                'get_items',
                entity_id=entity,
            )
            approved_operations = {
                '==': '==' in filter_by,
                '!=': '!=' in filter_by,
                '<': '<' in filter_by,
                '>': '>' in filter_by,
                'contains': 'contains' in filter_by,
            }

            conditions = {
                'has_items': bool(todo_items[entity]['items']),
                'has_filter': bool(filter_by) & any(approved_operations.values()),
            }

            if all(conditions.values()):
                todo_items = pd.DataFrame.from_dict(todo_items[entity]['items'])
                if 'due_datetime' in todo_items.columns:
                    todo_items['due_datetime'] = pd.to_datetime(todo_items['due_datetime'], format='ISO8601',
                                                                utc=True)
                else:
                    todo_items['due_datetime'] = pd.to_datetime(todo_items['due'], format='ISO8601', utc=True)

                try:
                    todo_items['due_datetime'] = todo_items['due_datetime'].dt.tz_convert('America/Chicago')
                except Exception as e:
                    self.log(f'Error in converting due_datetime: {e}', level="ERROR")
                    self.log(f"todo_items['due_datetime']:\n{todo_items['due_datetime'].to_markdown()}", level="ERROR")

                todo_items['due_datetime'] = todo_items['due_datetime'].dt.tz_localize(None)
                todo_items['due_datetime'] = todo_items['due_datetime'].apply(lambda x: pd.Timestamp(x))
                todo_items = todo_items.drop(columns=['due']) if 'due' in todo_items.columns else todo_items
                try:
                    todo_items = todo_items.query(filter_by)
                except Exception as e:
                    self.log(f'Error in query: {filter_by}', level="ERROR")
                    self.log(f'Error: {e}', level="ERROR")

                todo_items = todo_items.to_dict(orient='records')
            else:
                todo_items = todo_items[entity]['items']

            return todo_items

        except Exception as e:
            self.log(f"Error fetching todo items for {entity_id}: {str(e)}", level="ERROR")
            return None

    def _get_last_updated(self, entity_id, device_state, timestamp_format='datetime', persist=False):
        # Reading SQL template and formatting it
        sql_path = '/conf/apps/automations/queries/get_last_updated.sql'
        with open(sql_path, 'r') as file:
            sql = file.read().format(entity_id=entity_id, device_state=device_state)

        # Executing SQL and preparing the timestamp result
        timestamp_result = self.get_data(sql=sql)
        timestamp_result['current_state'] = timestamp_result['state']
        timestamp_result = timestamp_result.set_index('state')

        timestamp_result['last_changed_ts'] = timestamp_result['last_changed_ts'].apply(update_timestamp)

        current_time = datetime.now(pytz.timezone('America/Chicago'))
        persist_result = False  # Default value
        # Handling persistence and timestamp calculations
        if persist:
            minimum_time = timestamp_result['last_changed_ts'].max()
            persist_result = timestamp_result.loc[
                (timestamp_result['current_state'] == device_state) &
                (timestamp_result['last_changed_ts'] == minimum_time)
                ]

            persist_result = not persist_result.empty

        # Retrieving the last updated timestamp for the given device state
        timestamp = timestamp_result.T.get(device_state, pd.Series({'last_changed_ts': 'unavailable'}))
        timestamp = timestamp['last_changed_ts'] if isinstance(timestamp, pd.Series) else timestamp

        # Handling unavailable timestamp
        if timestamp == 'unavailable':
            self.log(f"get_last_updated()---No results found for {entity_id} and {device_state}\n\n{sql}",
                     level='DEBUG')
            timestamp = current_time - timedelta(days=1)  # Fallback to yesterday's date

        # Formatting the return value based on timestamp_format
        if timestamp_format == 'timedelta':
            timestamp = pd.Timedelta(current_time - timestamp)

        return (timestamp, persist_result) if persist else timestamp

    def _has_been_active(self, return_result='all', active_type='less than', time_threshold=10, still_state=True,
                         persist=True,
                         **matching_entities):
        results = {}
        state_map = {
            'opening': 'open',
            'closing': 'closed',
        }
        if 'filter_by' not in matching_entities:
            matching_entities.update({'filter_by': None})

        if 'get_attribute' not in matching_entities:
            matching_entities.update({'get_attribute': 'timedelta'})

        if isinstance(matching_entities['get_attribute'], str) and matching_entities['get_attribute'] != 'timedelta':
            matching_entities.update({'get_attribute': ['timedelta', matching_entities['get_attribute']]})

        elif isinstance(matching_entities['get_attribute'], list) and \
                'timedelta' not in matching_entities['get_attribute']:
            matching_entities.update({'get_attribute': ['timedelta', *matching_entities['get_attribute']]})

        def perform_check(entities_dict=None, **matching_entities):
            last_active_times = self.get_matching_entities(**matching_entities, persist=persist)
            last_active_times = last_active_times if last_active_times else {}

            for device, last_active in last_active_times.items():

                result = False
                device_state = state_map.get(matching_entities['device_state'], matching_entities['device_state'])
                current_state = last_active['state'] == device_state

                if active_type == 'less than' and last_active['timedelta'].total_seconds() < time_threshold:
                    if persist:
                        result = last_active['persist']
                    else:
                        result = True if not still_state else current_state

                elif active_type == 'greater than' and last_active['timedelta'].total_seconds() > time_threshold:
                    if persist:
                        result = last_active['persist']
                    else:
                        result = True if not still_state else current_state

                if device not in results:
                    if return_result == 'results':
                        results.update({device: {matching_entities['device_state']: result}})
                    else:

                        results.update({device: result})

                elif return_result == 'all':
                    results[device] = all((results[device], result))

                elif return_result == 'any':
                    results[device] = any((results[device], result))

                elif return_result == 'results':
                    results[device].update({matching_entities['device_state']: result})

            return results

        if isinstance(matching_entities['device_state'], list):
            device_states = matching_entities['device_state']
            matching_entities.pop('device_state')
        else:
            device_states = [matching_entities.pop('device_state')]

        for device_state in device_states:
            matching_entities.update({'device_state': device_state})
            results = perform_check(**matching_entities)

        if return_result == 'all':
            return all(list(results.values())) if results else False

        elif return_result == 'any':
            return any(list(results.values())) if results else False

        elif return_result == 'results':
            return results

    def _get_entity_attribute(self, entity_id, attribute_name, **kwargs):
        entity_info = self.get_state(entity_id, attribute="all")
        domain = entity_id.split('.')[0]

        # Check if the entity exists and has the requested attribute
        if entity_info is None:
            self.log(f"Entity {entity_id} not found.", level="ERROR")
            return None

        elif attribute_name == 'events' and domain == 'calendar':
            return self._get_calendar_events(
                entity_id=entity_id,
                start_date_time=kwargs.get('start_date_time'),
                end_date_time=kwargs.get('end_date_time'),
                duration=kwargs.get('duration'),
                filter_by=kwargs.get('filter_by'),
            )

        elif attribute_name == 'items' and domain == 'todo':
            return self._get_todo_items(
                entity_id=entity_id,
                filter_by=kwargs.get('filter_by'),
            )

        elif attribute_name == 'datetime' or attribute_name == 'timedelta':
            time_result, persist_result = self._get_last_updated(
                entity_id,
                kwargs.get('device_state'),
                timestamp_format=attribute_name,
                persist=True
            )
            return time_result, persist_result

        elif attribute_name == 'area':
            for area in self.areas:
                if area in entity:
                    return area

        elif "attributes" in entity_info and attribute_name in entity_info["attributes"]:
            return entity_info["attributes"][attribute_name]

        else:
            return None

    def _fire_results_to_event(self, result, fire_event):
        if fire_event:
            self.fire_event(fire_event['event_name'], f'{result}')

    def get_matching_entities(self, pattern=None, area=None, domain=None, pattern_overwrite=None, get_attribute=None,
                              device_state=None, device_state_query=None, device_output_type=None, agg_func=None,
                              agg_rank=None, troubleshoot=False, persist=False, unpack_single_entity=False,
                              exclude_pattern=None, exclude_pattern_overwrite=None, has_been_active_params=None,
                              filter_by='state', filter_by_and_or='or', exclude_manual_entities=None,
                              include_manual_entities=None, include_only=False, start_date_time=None,
                              end_date_time=None,
                              duration=None, fire_event=None, **kwargs):

        # Clean up kwargs. Remove any keys that are not valid
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith('_')}

        exclude_entities = self.get_matching_entities(
            pattern=exclude_pattern,
            pattern_overwrite=exclude_pattern_overwrite,
            area=area,
            domain=domain,
        ) if exclude_pattern is not None or exclude_pattern_overwrite is not None else {}

        patterns = []
        agg_func = agg_func if isinstance(agg_func, dict) else {'state': agg_func} if agg_func else None
        original_pattern = pattern.copy() if isinstance(pattern, list) else pattern
        original_pattern_overwrite = pattern_overwrite.copy() if isinstance(pattern_overwrite,
                                                                            list) else pattern_overwrite
        include_manual_entities = [] if include_manual_entities is None else include_manual_entities
        include_manual_entities = include_manual_entities if isinstance(include_manual_entities, list) else [
            include_manual_entities]

        if fire_event and (not isinstance(fire_event, dict) or 'event_name' not in fire_event):
            raise ValueError('fire_event must be a dictionary with event_name and event_info keys')

        matched_entities = self._build_matching_entities(  # Get List of all matching entities
            pattern=pattern,
            pattern_overwrite=pattern_overwrite,
            area=area,
            domain=domain,
            include_manual_entities=include_manual_entities,
            exclude_manual_entities=exclude_manual_entities,
            exclude_entities=exclude_entities,
            include_only=include_only,
        )

        if get_attribute:
            results = {}
            get_attribute = get_attribute if isinstance(get_attribute, list) else [get_attribute]
            for entity in set(matched_entities):
                for i, attr in enumerate(get_attribute):

                    if has_been_active_params:
                        attr_response = self._get_entity_attribute(
                            entity_id=entity,
                            attribute_name=attr,
                            start_date_time=start_date_time,
                            end_date_time=end_date_time,
                            duration=duration,
                            filter_by=filter_by,
                            device_state=device_state,
                            device_state_query=device_state_query,
                            has_been_active_params=has_been_active_params,
                        )
                    else:
                        attr_response = self._get_entity_attribute(
                            entity_id=entity,
                            attribute_name=attr,
                            start_date_time=start_date_time,
                            end_date_time=end_date_time,
                            duration=duration,
                            filter_by=filter_by,
                            device_state=device_state,
                            persist=persist,
                            device_state_query=device_state_query,
                        )

                    if i == 0:
                        if isinstance(attr_response, tuple):
                            results.update({entity: {attr: attr_response[0]}})
                            results[entity].update({'persist': attr_response[1]})
                        else:
                            results.update({entity: {attr: attr_response}})
                    else:
                        if isinstance(attr_response, tuple):
                            results[entity].update({attr: attr_response[0]})
                            results[entity].update({'persist': attr_response[1]})
                        else:
                            results[entity].update({attr: attr_response})

                results[entity].update({'state': self.get_state(entity)})

        else:
            results = {}
            for entity in set(matched_entities):
                results.update({entity: self.get_state(entity)})

        if has_been_active_params and (get_attribute == 'has_been_active' or (
                isinstance(get_attribute, list) and 'has_been_active' in get_attribute)):
            if 'active_type' not in has_been_active_params:
                self.log(f'still_state, persist, active_type, and time_threshold are the required parameters')
                raise ValueError('active_type must be specified in has_been_active_params')

            elif 'time_threshold' not in has_been_active_params:
                self.log(f'still_state, persist, active_type, and time_threshold are the required parameters')
                raise ValueError('time_threshold in minutes must be specified in has_been_active_params')

            has_been_active_results = self._has_been_active(
                return_result='results',  # This is fixed to get the results for all matching entities
                **has_been_active_params,
                # These are the main parameters for has_been_active
                # active_type='less than',
                # time_threshold=10,
                # still_state=True,
                area=area,
                domain=domain,
                pattern=original_pattern,
                pattern_overwrite=original_pattern_overwrite,
                device_state=device_state,
            )

            for entity, value in has_been_active_results.items():
                results[entity].update({'has_been_active': value})

        if get_attribute and 'calendar' not in domain and 'todo' not in domain:
            na = ['unavailable', 'unknown', 'None', None]
            if filter_by == 'state':
                if isinstance(device_state, list):
                    results = {entity: value for entity, value in results.items() if value['state'] in device_state}

                elif isinstance(device_state, str) or (isinstance(device_state, str) and is_numeric(device_state)):
                    results = {entity: value for entity, value in results.items() if value['state'] == device_state}

            elif filter_by == 'has_been_active':
                results = {entity: value for entity, value in results.items()
                           if any([val for d_state, val in value['has_been_active'].items()])}

            elif filter_by is not None:
                new_results = {}
                if isinstance(device_state, list):
                    for entity, value in results.items():
                        if isinstance(value[filter_by], list):
                            val = value[filter_by]
                            if filter_by_and_or == 'and':
                                has_match = len(set(device_state).intersection(set(val))) == len(device_state)
                                if has_match:
                                    new_results.update({entity: value})
                            elif filter_by_and_or == 'or':
                                has_match = set(device_state).intersection(set(val))
                                if has_match:
                                    new_results.update({entity: value})

                        elif value[filter_by] in device_state:
                            new_results.update({entity: value})

                elif isinstance(device_state, str) or (isinstance(device_state, str) and is_numeric(device_state)):
                    for entity, value in results.items():
                        if isinstance(value[filter_by], list):
                            val = value[filter_by]
                            if device_state in val:
                                new_results.update({entity: value})

                        elif value[filter_by] == device_state:
                            new_results.update({entity: value})

                        elif device_state in na and value[filter_by] in na:
                            new_results.update({entity: value})

                results = new_results

        elif not get_attribute:
            if filter_by == 'state' and device_state is not None:
                if isinstance(device_state, list):
                    results = {entity: value for entity, value in results.items() if value in device_state}

                elif isinstance(device_state, str) or (isinstance(device_state, str) and device_state.isnumeric()):
                    results = {entity: value for entity, value in results.items() if value == device_state}

        if device_output_type and get_attribute:
            if device_output_type == 'numeric':
                results = {entity: value for entity, value in results.items() if is_numeric(value['state'])}

            elif device_output_type == 'string':
                results = {entity: value for entity, value in results.items() if not is_numeric(value['state'])}

        elif device_output_type:
            if device_output_type == 'numeric':
                results = {entity: value for entity, value in results.items() if is_numeric(value)}

            elif device_output_type == 'string':
                results = {entity: value for entity, value in results.items() if not is_numeric(value)}

        if exclude_entities:
            results = {entity: value for entity, value in results.items() if entity not in exclude_entities}

        if exclude_manual_entities:
            results = {entity: value for entity, value in results.items() if entity not in exclude_manual_entities}

        if not results:
            self._fire_results_to_event(results, fire_event)
            return results

        if get_attribute and 'events' in get_attribute:
            calendar_events = {key: value['events'] for key, value in results.items() if 'calendar' in key}
        else:
            calendar_events = None

        results_df = pd.DataFrame.from_dict(results, orient='index', columns=None if get_attribute else ['state'])
        results_df = results_df.sort_index(ascending=False)

        try:
            # Replace 'unavailable' with np.nan
            for col in results_df.columns:
                if results_df[col].dtype == 'bool':
                    results_df[col] = results_df[col].astype('boolean')

                results_df.loc[results_df[col] == 'unavailable', col] = np.nan

        except Exception as e:
            self.log(f"Error in replacing 'unavailable' with np.nan: {e}")
            self._fire_results_to_event(results, fire_event)
            return results

        # Convert all columns to numeric if possible
        for col in results_df.columns:
            if results_df[col].dtype == 'object' and all([is_numeric(x) for x in results_df[col]]):
                results_df[col] = pd.to_numeric(results_df[col], errors='coerce')

        results_df = results_df.query(device_state_query) if device_state_query is not None else results_df
        # Name Index as Entity ID
        results_df.index.name = 'entity_id'

        results = results_df.to_dict(orient='index')
        results = {entity_id: results[entity_id]['state'] for entity_id in results} if not get_attribute else results

        index = None if 'index' not in kwargs else kwargs['index']
        if agg_func and not results_df.empty:
            agg_results = {}
            num_agg = len(agg_func)
            if isinstance(agg_func, dict):
                for func, agg in agg_func.items():
                    if index:
                        if num_agg == 1:
                            self._fire_results_to_event(results_df.groupby(index).agg({func: agg})[func].to_dict(),
                                                        fire_event)
                            return results_df.groupby(index).agg({func: agg})[func].to_dict()
                        else:
                            agg_results.update({func: results_df.groupby(index).agg({func: agg})[func].to_dict()})
                    else:
                        if num_agg == 1:
                            self._fire_results_to_event(results_df.agg({func: agg}).to_dict()['state'], fire_event)
                            return results_df.agg({func: agg}).to_dict()['state']
                        else:
                            agg_results.update(results_df.agg({func: agg}).to_dict())
                self._fire_results_to_event(agg_results, fire_event)
                return agg_results

        elif agg_rank and not results_df.empty:
            num_agg = len(agg_rank)
            if isinstance(agg_rank, dict):
                for func, agg in agg_rank.items():

                    if index:
                        results_df[f'{func}_rank'] = results_df.groupby(index)[func].rank(**agg)

                    else:
                        results_df[f'{func}_rank'] = results_df[func].rank(**agg)

                    tmp_df = results_df.query(f"{func}_rank==1").drop(columns=[f'{func}_rank'])

                    agg_rank[func] = tmp_df.to_dict(orient='index')
                if num_agg == 1:
                    agg_rank = agg_rank['state']
                    keys = list(agg_rank.keys())[0]
                    agg_rank[keys] = agg_rank[keys].pop('state')
            self._fire_results_to_event(agg_rank, fire_event)
            return agg_rank
        elif unpack_single_entity and len(results_df) == 1:
            (entity_id, entity_config), = results.items() if results else results_df.to_dict(orient='index').items()
            fire_results_to_event((entity_id, entity_config), fire_event)
            return entity_id, entity_config

        elif unpack_single_entity and len(results_df) > 1:
            self.log(
                f"""
                    get_matching_entities()---Multiple entities found for {pattern} and {area} and {domain}:

                    {results_df.to_string()}
                """,
                level='DEBUG',
            )
            self._fire_results_to_event(results, fire_event)
            return results
        else:
            if calendar_events:
                for entity, events in calendar_events.items():
                    results[entity].update({'events': events})

            self._fire_results_to_event(results, fire_event)
            return results

    def command_matching_entities(self, hacs_commands, pattern=None, area=None, domain=None, pattern_overwrite=None,
                                  get_attribute=None, device_state=None, device_state_query=None,
                                  device_output_type=None,
                                  troubleshoot=False, persist=False, exclude_pattern=None,
                                  exclude_pattern_overwrite=None,
                                  has_been_active_params=None, filter_by='state', filter_and_or='or',
                                  exclude_manual_entities=None, include_manual_entities=None, include_only=False,
                                  delay_execution=None, wait_until=None, fire_event=None, wait_until_timeout=60,
                                  **kwargs):

        def delay_func(delay_execution, **kwargs):
            self.run_in(self.command_matching_entities, delay_execution, **kwargs)

        def fire_results_to_event(result, fire_event):
            if fire_event:
                self._fire_results_to_event(fire_event['event_name'], {'info': result})

        if delay_execution:
            delay_func(delay_execution,
                       hacs_commands=hacs_commands,
                       pattern=pattern,
                       area=area,
                       domain=domain,
                       pattern_overwrite=pattern_overwrite,
                       get_attribute=get_attribute,
                       device_state=device_state,
                       device_state_query=device_state_query,
                       device_output_type=device_output_type,
                       troubleshoot=troubleshoot,
                       persist=persist,
                       exclude_pattern=exclude_pattern,
                       exclude_pattern_overwrite=exclude_pattern_overwrite,
                       has_been_active_params=has_been_active_params,
                       filter_by=filter_by,
                       filter_and_or=filter_and_or,
                       exclude_manual_entities=exclude_manual_entities,
                       include_manual_entities=include_manual_entities,
                       include_only=include_only,
                       **kwargs
                       )
            return

        elif wait_until:
            pass
            self.log(f"NOT IMPLEMENTED YET: wait_until={wait_until} and wait_until_timeout={wait_until_timeout}")
            # 
            # wait_until_info = task.wait_until(
            #     state_trigger=wait_until,
            #     timeout=wait_until_timeout,
            # )
            # 
            # if wait_until_info['trigger_type'] == 'timeout':
            #     self.log(f'wait_until timed out after {wait_until_timeout} seconds')
            #     return
            # 
            # else:
            #     self.log(f'wait_until triggered')

        # Clean up kwargs. Remove any keys that are not valid for the service call
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith('_')}

        if isinstance(hacs_commands, list):
            hacs_commands = {command: kwargs for command in hacs_commands}

        elif isinstance(hacs_commands, str):
            hacs_commands = {hacs_commands: kwargs}

        matching_entities = self.get_matching_entities(
            pattern=pattern,
            pattern_overwrite=pattern_overwrite,
            area=area,
            domain=domain,
            get_attribute=get_attribute,
            device_state=device_state,
            device_state_query=device_state_query,
            device_output_type=device_output_type,
            troubleshoot=troubleshoot,
            persist=persist,
            exclude_pattern=exclude_pattern,
            exclude_pattern_overwrite=exclude_pattern_overwrite,
            exclude_manual_entities=exclude_manual_entities,
            include_manual_entities=include_manual_entities,
            include_only=include_only,
            has_been_active_params=has_been_active_params,
            filter_by=filter_by,
        )

        output = {}
        for command, hacs_kwargs in hacs_commands.items():
            entities_by_domain = {}
            for entity, value in matching_entities.items():
                c_domain = entity.split('.')[0] if 'domain' not in hacs_kwargs else hacs_kwargs['domain']
                if c_domain not in entities_by_domain:
                    entities_by_domain.update({c_domain: [entity]})
                else:
                    entities_by_domain[c_domain].append(entity)

                hacs_kwargs.pop('domain') if 'domain' in hacs_kwargs else None
            for domain, entities in entities_by_domain.items():
                special_cases = {}
                if 'set' in command:
                    for ent in entities:
                        c = command.replace('set', '').strip('_') if 'preset' not in command \
                            else command.replace('set', '').replace('pre', 'preset').strip('_')

                        special_case = {
                            'has_attribute': (bool(self._has_attribute(ent, [f'{c}s', f'available_{c}s'], []))),
                            'not_available': (
                                    hacs_kwargs.get(c, None) not in
                                    self._has_attribute(ent, [f'{c}s', f'available_{c}s'], []))
                        }

                        if all(special_case.values()):
                            special_cases.update({ent: command})

                    for ent in special_cases:
                        self.log(f"Removing {special_cases[ent]} with {hacs_kwargs}")
                        entities.remove(ent)

                try:
                    if troubleshoot:
                        self.log(
                            f"Calling service with\n"
                            f"domain: {domain}, \n"
                            f"command: {command}, \n"
                            f"entity_id: {entities}, \n"
                            f"kwargs: {hacs_kwargs}"
                        )
                    else:
                        self.call_service(service=f'{domain}/{command}', entity_id=entities, **hacs_kwargs)
                except Exception as e:
                    self.log(f"""
                    Error calling {command} on {entities} with {hacs_kwargs}
                    {e}
                    """)
                    output[domain].update({command: 'Error', 'entities': entities})

                # self.sleep(1)
                if entity in output and command not in output[entity]:
                    output[domain].update({command: 'Success', 'entities': entities, **hacs_kwargs})
                else:
                    output.update({domain: {command: 'Success', 'entities': entities, **hacs_kwargs}})

        self._fire_results_to_event(output, fire_event)
        return output

    def _has_attribute(self, entity_id, attributes, fall_back):
        """
        Check if the entity has the given attribute(s).
        :param entity_id:
        :param attributes:
        :param fall_back:
        :return:
        """
        for attribute in attributes:
            try:
                result = self.get_state(entity_id, attribute=attribute)
                if result is not None:
                    return result
            except:
                pass
        return fall_back

    def find_rooms_with_occupancy(self, app):
        """
        Find all rooms with occupancy sensors that are currently on.
        :param app:
        :return:
        """
        rooms_with_occupancy = self.get_matching_entities(
            domain='binary_sensor',
            pattern='occupancy',
            get_attribute='area',
            agg_func='count',
            index='area',
            device_state='on',
        )
        add_shared_rooms = {}
        for room in rooms_with_occupancy:
            rooms = find_associated_rooms(room, app)
            for r in rooms:
                if r not in rooms_with_occupancy:
                    add_shared_rooms.update({r: {'off': 1}})

        rooms_with_occupancy.update(add_shared_rooms)

        return list(rooms_with_occupancy.keys())

    def calculate_individual_score(self, current_value, optimal_value, condition, **kwargs):

        if condition == 'greater':
            # Define steepness for 'greater' condition
            steepness = 1.5 if 'greater' not in kwargs else kwargs['greater']['steepness']
            if optimal_value != 0:
                return (current_value / optimal_value) ** steepness
            else:
                return 1 if current_value >= 0 else 0

        elif condition == 'lower':
            # Define scaler for 'lower' condition
            scaler = 10 if 'lower' not in kwargs else kwargs['lower']['scaler']
            if optimal_value != 0:
                return 1 / (1 + np.exp(scaler * ((current_value / optimal_value) - 1)))
            else:
                return 1 if current_value == 0 else 0

        elif condition == 'range':
            if isinstance(optimal_value, tuple) and len(optimal_value) == 2:
                lower_limit, upper_limit = optimal_value
                middle_value = (lower_limit + upper_limit) / 2

                if lower_limit <= current_value <= upper_limit:
                    # Define inside_scaler for 'range' condition when inside the range
                    inside_scaler = -2 if 'range' not in kwargs else kwargs['range']['inside_scaler']
                    normalized_value = 2 * (current_value - lower_limit) / (upper_limit - lower_limit)
                    if normalized_value <= 1:
                        score = 0.5 + 0.5 * np.exp(inside_scaler * (1 - normalized_value))
                    else:
                        score = 0.5 + 0.5 * np.exp(inside_scaler * (normalized_value - 1))
                else:
                    # Define outside_scaler for 'range' condition when outside the range
                    outside_scaler = -5 if 'range' not in kwargs else kwargs['range']['outside_scaler']
                    distance = min(np.abs(current_value - lower_limit), np.abs(current_value - upper_limit))
                    score = 0.5 * np.exp(outside_scaler * distance / (upper_limit - lower_limit))

                return score
            else:
                return 0
        else:
            return 0

    def extract_between(self, text, start, end):
        pattern = rf"{re.escape(start)}_(.+?)_{re.escape(end)}"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        return None  # or some default value if not found

    def log_info(self, log_config, log_func, app, message, function='all', level='INFO', log_room='all',
                 notify_device=None, notify_light=None, notify_light_params=None, notify_tts=None,
                 chat_gpt_executor=None, chat_gpt_text_to_speech=None, chat_gpt=None, chat_gpt_voice=None,
                 room_override=None, title=None, tts_override=False, tts_kwargs=None, metadata_tts=None,
                 chat_gpt_voice_replace=False, modify_metadata=None, notify_device_action=None,
                 notify_device_timeout=30, notify_device_wait_for_response=False, **kwargs):

        room = log_config[app]['logging']['room'] if app in log_config else 'all'
        config_level = log_config[app]['logging']['level'] if app in log_config else 'INFO'
        config_function = log_config[app]['logging']['function'] if app in log_config else 'all'

        if isinstance(log_room, str) and log_room == room or room == 'all':
            if function == config_function or config_function == 'all':
                if (level == config_level or (config_level == 'DEBUG' and level == 'INFO')) or config_level == 'all':
                    log_func(f'{app}.{function}: {message}', level=level)
                else:
                    pass
            else:
                pass
        elif isinstance(log_room, list) and room in log_room or room == 'all':
            if function == config_function or config_function == 'all':
                if (level == config_level or (config_level == 'DEBUG' and level == 'INFO')) or config_level == 'all':
                    log_func(f'{app}.{function}: {message}', level=level)
                else:
                    pass
            else:
                pass
        else:
            pass

    def master_automation_logic(self, *args, **kwargs):
        """
        Master function for automation logic. This turns on a boolean, runs a set of commands, waits, then runs a set of
        final commands. This is useful for automations that need to run a set of commands, wait, then run a set of final

        """

        app_name = kwargs.pop('app_name')
        master_name = kwargs.pop('master_name')
        boolean_checks = kwargs.pop('boolean_checks')
        commands = kwargs.pop('commands')
        final_commands = kwargs.pop('final_commands')
        default_wait = kwargs.pop('default_wait')
        task_id = kwargs.pop('task_id')

        if self.should_debounce(f"{master_name}"):
            return

        self.log(
            msg=f"Starting {master_name.replace('_', ' ').title()} Cycle",
            level='INFO',
        )

        if all(boolean_checks.keys()):

            # # Turn on master is running boolean
            # response = self.command_matching_entities(
            #     hacs_commands='turn_on',
            #     domain='input_boolean',
            #     pattern_overwrite=f"input_boolean.{app_name}_{task_id}_master_is_running",
            #     device_state='off'
            # )

            if 'hours' in kwargs:
                wait_for = kwargs['hours'] * 3600

            elif 'minutes' in kwargs:
                wait_for = kwargs['minutes'] * 60

            elif 'seconds' in kwargs:
                wait_for = kwargs['seconds']

            else:
                wait_for = default_wait

            for command in commands:
                response = self.command_matching_entities(**command)

            if final_commands:
                for command in final_commands:
                    self.run_in(self.command_matching_entities, wait_for, **command)

            # # Turn off master is running boolean
            # response = self.command_matching_entities(
            #     hacs_commands='turn_off',
            #     domain='input_boolean',
            #     pattern_overwrite=f'{app_name}_{master_name}_master_is_running$',
            #     device_state='on'
            # )

    def generate_patterns(self, single_pattern, single_area, single_domain):
        single_pattern = ".*" if single_pattern is None else single_pattern
        pattern_areas = "*" if single_area is None else single_area

        if single_domain is None:
            return f'.{pattern_areas}_.{single_pattern}'

        elif single_domain and single_area is None and single_pattern == ".*":
            return f'{single_domain}.*'

        else:
            return fr'{single_domain}.{pattern_areas}_{single_pattern}'

    def trigger_event_recorder(self, *args, **kwargs):

        # Start recording events
        duration = kwargs.get('duration', 30)  # seconds
        recording_key = kwargs.get('recording_key', 'default')
        regex_filter = kwargs.get('regex_filter', None)

        if self.should_debounce(recording_key):
            return

        self.record_events(duration, recording_key, regex_filter)

    def record_events(self, duration, recording_key, regex_filter=None):
        self.log("Starting to record events.")

        # Initialize a list to hold recorded events
        self.recorded_events = []

        # Capture the start time
        start_time = datetime.now()

        # Set up a listener for all state changes
        self.event_listener = self.listen_state(
            self.capture_event,
            regex_filter=regex_filter
        )

        # Schedule stopping the recording after the specified duration
        self.run_in(
            self.stop_recording,
            delay=duration,
            start_time=start_time,
            recording_key=recording_key,
        )

    def capture_event(self, entity, attribute, old, new, **kwargs):
        # Record the event details
        regex_filter = kwargs.get('regex_filter', [None])
        regex_filter = regex_filter if isinstance(regex_filter, list) else [regex_filter]

        event_details = {
            "entity": entity,
            "attribute": attribute,
            "old": old,
            "new": new,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        for regex in regex_filter:
            if regex is not None:
                self.log(f"Checking {entity} against {regex}")
                if re.match(regex, entity):
                    self.recorded_events.append(event_details)
                    return
            else:
                self.recorded_events.append(event_details)

    def stop_recording(self, **kwargs):

        recording_key = kwargs.get('recording_key', 'default')

        # Stop listening to state changes
        self.cancel_listen_state(self.event_listener)

        # Log the recorded events or process them as needed
        self.log(f"Stopped recording events. Recorded {len(self.recorded_events)} events.")

        # Optionally, return the recorded events if this method is used in a context
        # where the return value can be captured and utilized
        self.recordings[recording_key] = self.recorded_events

        data = pd.DataFrame.from_records(self.recorded_events)
        # Convert the time column to a datetime object and adjust for the local timezone
        data['time'] = pd.to_datetime(data['time'])
        data['time'] = data['time'].dt.tz_localize('UTC').dt.tz_convert('America/Chicago')

        data['recording_key'] = recording_key
        data.to_sql('activity_tracking', con=self.home_engine, if_exists='append', index=False)

        return self.recorded_events


def update_timestamp(x, return_result='datetime'):
    # Separate into seconds and fractional seconds
    timestamp_seconds = int(x)
    timestamp_fractional = x - timestamp_seconds

    # Convert to datetime
    converted_seconds = pd.to_datetime(timestamp_seconds, unit='s', utc=True)
    converted_fractional = pd.to_timedelta(timestamp_fractional, unit='s')

    # Combine
    converted = converted_seconds + converted_fractional

    # Convert to local timezone
    local_converted = converted.tz_convert('America/Chicago')
    if return_result == 'datetime':
        return pd.to_datetime(local_converted)

    elif return_result == 'timedelta':
        current_time = datetime.now(pytz.timezone('America/Chicago'))
        return pd.Timedelta(current_time - local_converted)


def parse_datetime_with_timezone(datetime_str, return_string=True, string_format='%Y-%m-%d %H:%M:%S'):
    """
    Parses a datetime string with timezone information.

    Args:
    datetime_str (str): A datetime string in the format 'YYYY-MM-DDTHH:MM:SS+HH:MM' or 'YYYY-MM-DDTHH:MM:SS-HH:MM'.

    Returns:
    datetime: The parsed datetime object adjusted for the timezone.
    """
    # Extract the datetime and timezone parts
    match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})([+-]\d{2}:\d{2})?", datetime_str)
    if match:
        datetime_part, tz_offset = match.groups()

        # Convert the datetime part to a datetime object
        if return_string:
            return datetime.strptime(datetime_part, string_format).strftime(string_format)

        else:
            return datetime.strptime(datetime_part, string_format)
    else:
        raise ValueError("Invalid datetime format")


def get_parsed_attribute(attribute):
    if isinstance(attribute, list):
        new_list = []
        for attr in attribute:
            if isinstance(attr, Enum):
                if attr.name is not None:
                    new_list.append(attr.name.lower())
                else:
                    new_list.append(None)

            elif isinstance(attr, str):
                new_list.append(attr.lower())
        return new_list

    elif isinstance(attribute, Enum):
        if attribute.name is not None and '|' in attribute.name:
            attribute = [attr.lower() for attr in attribute.name.split('|')]
            return attribute

        elif attribute.name is not None:
            return attribute.name.lower()

        else:
            return None
    else:
        return str(attribute)


def is_numeric(value):
    """
    Checks if the given value is numeric (int, float, or a string representing a number).
    """
    if isinstance(value, Number) or value is None:
        return True
    if isinstance(value, str):
        try:
            float(value)
            return True
        except ValueError:
            return False
    return False


def extract_between(text, start, end):
    pattern = rf"{re.escape(start)}_(.+?)_{re.escape(end)}"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None  # or some default value if not found


