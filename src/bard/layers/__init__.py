from . import attention, decoder, embedding, encoder, masking

__all__ = ['attention', 'decoder', 'embedding', 'encoder', 'masking', 'underlying_value']

def underlying_value(val, type):
   """returns the underlying value of `val`, as `type`. Valid type conversion must exist, `type`
   must be callable."""
   return val if isinstance(val, type) else type(val)
