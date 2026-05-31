"""Constants for FileEditTool."""

# Tool name
FILE_EDIT_TOOL_NAME = "edit_file"

# Maximum file size for editing (1 GiB)
MAX_EDIT_FILE_SIZE = 1024 * 1024 * 1024

# Error messages
FILE_UNEXPECTEDLY_MODIFIED_ERROR = (
    "File has been unexpectedly modified. Read it again before attempting to write it."
)

FILE_NOT_FOUND_CWD_NOTE = "File does not exist. Make sure the path is relative to the current working directory:"

# Additional error codes
ERROR_JUPYTER_NOTEBOOK = "Cannot edit Jupyter notebooks. Use NotebookEditTool."
ERROR_NOT_READ_FIRST = "File has not been read yet. Read it first before editing."
ERROR_MODIFIED_SINCE_READ = "File has been modified since read. Read it again before editing."
ERROR_PARTIAL_VIEW = "File was only partially read. Read the full file before editing."
