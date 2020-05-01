import sys
print(sys.argv)
sys.argv[0] = sys.argv[0] + 'test'
print(sys.argv)
