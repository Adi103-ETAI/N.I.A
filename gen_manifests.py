import os
import yaml

manifests = [
    {'name': 'read_file', 'scope': 'read_only', 'reversible': True, 'description': 'Reads a file from the host filesystem.', 'timeout': 30},
    {'name': 'ls_dir', 'scope': 'read_only', 'reversible': True, 'description': 'Lists directory contents.', 'timeout': 30},
    {'name': 'find_by_name', 'scope': 'read_only', 'reversible': True, 'description': 'Finds a file by name.', 'timeout': 60},
    {'name': 'grep_search', 'scope': 'read_only', 'reversible': True, 'description': 'Searches file contents using grep.', 'timeout': 60},
    {'name': 'write_file', 'scope': 'write', 'reversible': False, 'description': 'Writes a new file to the host filesystem.', 'timeout': 60},
    {'name': 'edit_file', 'scope': 'write', 'reversible': False, 'description': 'Edits an existing file.', 'timeout': 60},
    {'name': 'mkdir_dir', 'scope': 'write', 'reversible': False, 'description': 'Creates a directory.', 'timeout': 30},
    {'name': 'run_in_sandbox', 'scope': 'execute', 'reversible': False, 'description': 'Runs bash/python/node commands inside the secure Docker sandbox.', 'timeout': 300},
    {'name': 'invoke_tara', 'scope': 'execute', 'reversible': False, 'description': 'Invokes the TARA technical executor subagent.', 'timeout': 300},
    {'name': 'invoke_iris', 'scope': 'execute', 'reversible': False, 'description': 'Invokes the IRIS vision subagent.', 'timeout': 120},
]

os.makedirs('src/capabilities/manifests', exist_ok=True)
for m in manifests:
    with open(f'src/capabilities/manifests/{m["name"]}.yaml', 'w') as f:
        yaml.dump(m, f)
print('Manifests created successfully.')
