import json

def parse_client_data(data):
    """
    Parse client data that may be a JSON string with escaped backslashes.
    Also cleans specific fields that require additional processing.
    
    Args:
        data: Input data which might be a string or already parsed dict
        
    Returns:
        dict: Parsed data as a dictionary with cleaned fields
        
    Raises:
        ValueError: If JSON parsing fails
    """
    # If data is already a dict, work with a copy to avoid modifying the original
    if isinstance(data, dict):
        parsed_data = data.copy()
    
    # If data is a string, clean and parse it
    elif isinstance(data, str):
        # First, clean double backslashes, then single backslashes
        # This addresses the escaping issue from Swift's Socket.IO implementation
        clean_data = data.replace("\\\\", "\\").replace("\\", "")
        
        try:
            parsed_data = json.loads(clean_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
    
    # If data is neither a dict nor a string, we can't handle it
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
    
    # Clean specific fields
    if "model_name" in parsed_data and parsed_data["model_name"] is not None:
        parsed_data["model_name"] = parsed_data["model_name"].replace("\\", "").strip()
    
    # Also clean session_id if present and not None
    if "session_id" in parsed_data and parsed_data["session_id"] is not None:
        parsed_data["session_id"] = parsed_data["session_id"].replace("\\", "").strip()
    
    return parsed_data