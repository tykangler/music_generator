def underlying_value(val, type):
   """returns the underlying value of `val`, as `type`. Valid type conversion must exist, `type`
   must be callable."""
   return val if isinstance(val, type) or val is None else type(val)
