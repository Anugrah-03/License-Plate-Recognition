import redis
import base64

from PIL import Image
from io import BytesIO

# Connect to Redis
r = redis.StrictRedis(host='localhost', port=6379, db=0)

# Open image and convert to base64
with Image.open('License-Plate-Recognition/detected_boxes/box_0.jpg') as image:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())

# Store in Redis
r.set('my_image_key', img_str)
# Fetch from Redis
img_str_from_redis = r.get('my_image_key')

# Convert base64 back to image
img_bytes = base64.b64decode(img_str_from_redis)
img = Image.open(BytesIO(img_bytes))
img.show()
