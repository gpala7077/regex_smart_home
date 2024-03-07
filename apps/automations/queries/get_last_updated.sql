SELECT
    state,
    -- last_changed/last_updated is not consistently filled in. Case statement ensures at least one of them is
    -- returned.
    MAX(COALESCE(last_changed_ts, last_updated_ts)) as last_changed_ts
FROM states
INNER JOIN states_meta ON states_meta.metadata_id = states.metadata_id
WHERE states_meta.entity_id = '{entity_id}'
GROUP BY state
