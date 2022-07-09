from .core import Cache


def set(args):
    cache = Cache()
    cache.set(key=args.key, value=args.value, expire=args.expire)

def delete(args):
    cache = Cache()
    cache.delete(key=args.key)