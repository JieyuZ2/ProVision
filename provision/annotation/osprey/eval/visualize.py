import os
import base64

def display_string(output: str):
    ''' Replace new line for html'''
    return output.replace('\n','<br>')

def display_list(output: list):
    ''' Displays list of strings in html'''
    return '<br><br>==<br>'.join(output).replace('\n','<br>')

def display_image(image_path: str, width=400, height=400):
    if not os.path.isfile(image_path):
        return None
    
    # If stored in web...
    if 'http' in image_path:
        return f'<img src="{image_path}" width="{width}" height="{height}" />'
    
    # Use local image
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return f'<img src="data:image/png;base64,{encoded_string}" width="{width}" height="{height}" />'  # Adjust width and format as needed