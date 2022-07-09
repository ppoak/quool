import argparse
from .tools.cli import set, delete



parser = argparse.ArgumentParser(description='PandasQuant CLI API')
subparser = parser.add_subparsers()

setter = subparser.add_parser('set', help='Set cache via pandasquant Cli')
setter.add_argument('-k', '--key', required=True, type=str, help='Key to the cache')
setter.add_argument('-v', '--value', required=True, help='Value to the cache')
setter.add_argument('-e', '--expire', default=None, help='Valid time, default to forever')
setter.set_defaults(func=set)

deleter = subparser.add_parser('delete', help='Delete cache via pandasquant Cli')
deleter.add_argument('-k', '--key', required=True, type=str, help='Key to the cache')
deleter.set_defaults(func=delete)

args = parser.parse_args()
args.func(args)
