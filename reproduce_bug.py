
try:
    import nia
    print("Imported nia successfully")
    try:
        nia.process_input
        print("nia.process_input is accessible via attribute access")
    except AttributeError:
        print("nia.process_input missing")

    # Now try to call demo() which uses process_input inside the module
    # This should fail if my hypothesis is correct
    print("Attempting to run a snippet of demo logic...")
    try:
        # We can't easily run the infinite loop of demo(), so we'll just check if the name is defined in the module's globals
        # by inspecting the function's closure or globals
        if 'process_input' in nia.demo.__globals__:
            print("process_input is in demo globals")
        else:
            print("process_input is NOT in demo globals (Bug confirmed)")
    except Exception as e:
        print(f"Error inspecting demo: {e}")

except Exception as e:
    print(f"Import failed: {e}")
