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
from croniter import croniter


# Base Home Assistant Class for Automations
class Base(hass.Hass):
    debounce_timers = {}
    debounce_period = timedelta(seconds=60)
    cron_job_funcs = {}

    def initialize(self):
        self.timezone = pytz.timezone(self.args.get('timezone', 'America/Chicago'))

    def should_debounce(self, debounce_key, debounce_period=None):
        """Check if the call should be debounced."""
        current_time = datetime.now()
        last_run = self.debounce_timers.get(debounce_key)
        debounce_period = debounce_period if debounce_period is not None else self.debounce_period

        if last_run is None:
            # If it's the first run, do not debounce
            self.debounce_timers[debounce_key] = current_time
            return False

        if current_time - last_run < debounce_period:
            # If we're within the debounce period, skip this call
            return True

        # If we're outside the debounce period, proceed with the call
        self.debounce_timers[debounce_key] = current_time
        return False

    def create_cron_jobs(self):
        """Create cron jobs."""
        cron_jobs = self.args.get('cron_job_schedule', {})

        for job, config in cron_jobs.items():
            run_immediately = config.pop('run_immediately', False)
            interval = config.pop('interval', 3600)
            app_func = config.pop('function', 'run_every')
            time_pattern = config.pop('time_pattern', '00:00:00')
            # Convert to time object
            time_pattern = datetime.strptime(time_pattern, '%H:%M:%S').time()
            time_pattern = datetime.combine(datetime.now(self.timezone).date(), time_pattern)

            if run_immediately:
                self.cron_job_funcs[job]()

            if app_func == 'run_hourly':
                self.run_hourly(self.cron_job_funcs[job], start=time_pattern, **config)

            elif app_func == 'run_every':
                self.run_every(self.cron_job_funcs[job], start=time_pattern, interval=interval, **config)

    def log_info(self, app, message, function='all', level='INFO', log_room='all',
                 notify_device=None, notify_light=None, notify_light_params=None, notify_tts=None,
                 title=None, notify_device_action=None, notify_device_timeout=30,
                 notify_device_wait_for_response=False, **kwargs):

        room = self.args.get('logging', {}).get('room', 'all')
        config_level = self.args.get('logging', {}).get('level', 'INFO')
        config_function = self.args.get('logging', {}).get('function', 'all')

        if isinstance(log_room, str) and log_room == room or room == 'all':
            if function == config_function or config_function == 'all':
                if (level == config_level or (config_level == 'DEBUG' and level == 'INFO')) or config_level == 'all':
                    self.log(f'{app}.{function}: {message}', level=level)
                else:
                    pass
            else:
                pass
        elif isinstance(log_room, list) and room in log_room or room == 'all':
            if function == config_function or config_function == 'all':
                if (level == config_level or (config_level == 'DEBUG' and level == 'INFO')) or config_level == 'all':
                    self.log(f'{app}.{function}: {message}', level=level)
                else:
                    pass
            else:
                pass
        else:
            pass
