from slowapi import Limiter
from slowapi.util import get_remote_address

# Initialize global rate limiter using client IP address
limiter = Limiter(key_func=get_remote_address)
