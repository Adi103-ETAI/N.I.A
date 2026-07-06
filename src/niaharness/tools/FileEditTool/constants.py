"""FileEditTool constants."""

FILE_EDIT_TOOL_NAME = "edit_file"
FILE_NOT_FOUND_CWD_NOTE = "File does not exist. Current working directory:"
FILE_UNEXPECTEDLY_MODIFIED_ERROR = (
    "File has been modified since last read, either by the user or by a linter. "
    "Read it again before attempting to write it."
)
MAX_EDIT_FILE_SIZE = 1024 * 1024 * 1024  # 1 GiB

# Error messages
ERROR_JUPYTER_NOTEBOOK = (
    "File is a Jupyter Notebook. Use the notebook_edit tool to edit this file."
)
ERROR_NOT_READ_FIRST = (
    "File has not been read yet. Read it first before writing to it."
)
ERROR_MODIFIED_SINCE_READ = FILE_UNEXPECTEDLY_MODIFIED_ERROR
