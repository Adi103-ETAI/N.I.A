import asyncio
import sys
import os

# Add project root to path to ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.events import get_event_bus
from src.core.logger import start_log_listener, setup_logger

async def main():
    # Setup basic logging to see the output
    # We init the logger first so the handler is set up to print to console
    setup_logger("TEST_BOT")
    
    # Activate the subscriber
    print("Initializing Log Listener...")
    start_log_listener()
    
    bus = get_event_bus()
    
    print("--- Starting Test ---")
    
    payload = {
        "level": "INFO", 
        "source": "TEST_BOT", 
        "message": "✅ SUCCESS: The Event Bus is carrying this message!"
    }
    
    try:
        print(f"Emitting payload: {payload['message']}")
    except UnicodeEncodeError:
        print(f"Emitting payload: {payload['message'].encode('ascii', 'replace').decode()}")
    # Emit the event
    await bus.emit("log:entry", payload)
    
    # Yield control to let the event loop process the emitted task
    await asyncio.sleep(0.5)
    
    print("--- Test Finished ---")

if __name__ == "__main__":
    asyncio.run(main())
