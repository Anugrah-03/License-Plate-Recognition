import redis

# Connect to the local Redis instance
client = redis.StrictRedis(host='localhost', port=6379, db=0)

# Set a key-value pair
client.set('my_key', 'my_value')

# Retrieve the value
value = client.get('my_key')
print(value.decode('utf-8'))  # Decode from bytes to string
