# utils for visualizing to html
import numpy as np
from PIL import Image
import pandas as pd
from functools import partial
import io
import base64
import re
import json
from tqdm import tqdm

def pil_to_html(img: Image.Image, max_width=300):
    
    if img is None:
        return ""
    
    if isinstance(img, str): # url
        if re.match(r"https?://", img):
            # with width
            return f'<img src="{img}" width="{max_width}" />'
        else: # local path
            img = Image.open(img)
    
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    # Display PIL image instead
    buf = io.BytesIO()
    img = img.convert('RGB')  # Convert to RGB for JPEG format
    img.thumbnail((max_width, max_width))
    
    img.save(buf, format='JPEG', quality=85)
    data = base64.b64encode(buf.getvalue()).decode('utf-8')  # Convert bytes to base64 string
    img_src = f"data:image/jpeg;base64,{data}"
    # return f'<a href="#lightbox" onclick="showLightbox(\'{img_src}\')"><img src="{img_src}" width="{max_width}" /></a>'
    return f'<img src="data:image/png;base64,{data}" />'

def str_to_html(s: str):
    if isinstance(s, dict):
        s = json.dumps(s, indent=2)
    s = s.replace('\n', '<br>')
    
    # Escape dollar signs that are used for LaTeX math mode
    s = re.sub(r'\$(.+?)\$', r'\\(\1\\)', s)
    
    return s

def convert_df_to_html(df: pd.DataFrame, image_keys: list[str]=None, text_keys: list[str]=None, 
                        max_rows=300,
                        max_width=300):
    if image_keys is None:
        image_keys = []
    if text_keys is None:
        text_keys = []
    
    
    # use image_keys and text_keys to apply the formatters
    formatters = {}
    for key in image_keys:
        formatters[key] = partial(pil_to_html, max_width=max_width)
    for key in text_keys:
        formatters[key] = str_to_html
    
    # tqdm.pandas()
    # html = df.progress_apply(lambda row: row.to_frame().T.to_html(escape=False, formatters=formatters, justify='center', col_space=10, header=False, index=False), axis=1)
    html = df.to_html(escape=False, formatters=formatters, justify='center', col_space=10)
    
    # CSS to make the table responsive and for the lightbox
    css = """
    <style>
        table {
            table-layout: fixed;
            width: 100%;
            border-collapse: collapse;
        }

        th, td {
            text-align: left;
            vertical-align: middle; /* Centers content vertically */
            overflow: hidden;
            text-overflow: ellipsis;
            word-wrap: break-word;
        }

        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }

        th:first-child, td:first-child {
            width: 2%;
            white-space: nowrap;
        }
        
        th:nth-child(2), td:nth-child(2) {
            width: 30%; /* Set the image column to cover 60% of the screen */
        }

        th:nth-child(3), td:nth-child(3) {
            width: 30%; /* Set the image column to cover 60% of the screen */
        }


        td {
            position: relative;
            overflow: hidden;
        }

        td img {
            max-height: 100%;
            max-width: 100%;
            object-fit: contain;
            display: block;
            margin: 0 auto;
            cursor: pointer;
        }
    </style>
    """

    # JavaScript for pagination
    js = f"""
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            const rowsPerPage = {max_rows};
            const table = document.querySelector('table');
            const rows = Array.from(table.querySelectorAll('tbody tr'));
            const totalRows = rows.length;
            const totalPages = Math.ceil(totalRows / rowsPerPage);
            let currentPage = 1;

            function showPage(page) {{
                const start = (page - 1) * rowsPerPage;
                const end = start + rowsPerPage;
                rows.forEach((row, index) => {{
                    row.style.display = (index >= start && index < end) ? '' : 'none';
                }});
                document.getElementById('prevButtonTop').disabled = page === 1;
                document.getElementById('nextButtonTop').disabled = page === totalPages;
                document.getElementById('prevButtonBottom').disabled = page === 1;
                document.getElementById('nextButtonBottom').disabled = page === totalPages;
                window.scrollTo(0, 0); // Scroll to top
            }}

            document.getElementById('prevButtonTop').addEventListener('click', function() {{
                if (currentPage > 1) {{
                    currentPage--;
                    showPage(currentPage);
                }}
            }});

            document.getElementById('nextButtonTop').addEventListener('click', function() {{
                if (currentPage < totalPages) {{
                    currentPage++;
                    showPage(currentPage);
                }}
            }});

            document.getElementById('prevButtonBottom').addEventListener('click', function() {{
                if (currentPage > 1) {{
                    currentPage--;
                    showPage(currentPage);
                }}
            }});

            document.getElementById('nextButtonBottom').addEventListener('click', function() {{
                if (currentPage < totalPages) {{
                    currentPage++;
                    showPage(currentPage);
                }}
            }});

            showPage(currentPage);
        }});
    </script>
    """

    # Pagination controls
    pagination_controls_top = """
    <div class="pagination">
        <button id="prevButtonTop" disabled>Previous</button>
        <button id="nextButtonTop">Next</button>
    </div>
    """
    
    pagination_controls_bottom = """
    <div class="pagination">
        <button id="prevButtonBottom" disabled>Previous</button>
        <button id="nextButtonBottom">Next</button>
    </div>
    """

    html = f"{css}\n{pagination_controls_top}\n{html}\n{pagination_controls_bottom}\n{js}"
    
    return html

def convert_df_to_html_v2(df: pd.DataFrame, image_keys=None, text_keys=None, max_rows=300, max_width=300):
    if image_keys is None:
        image_keys = []
    if text_keys is None:
        text_keys = []
    
    html_pages = []
    total_rows = len(df)
    num_pages = (total_rows // max_rows) + (1 if total_rows % max_rows != 0 else 0)
    
    for page_num in range(num_pages):
        start_idx = page_num * max_rows
        end_idx = min(start_idx + max_rows, total_rows)
        page_df = df.iloc[start_idx:end_idx]
        page_html = ''
        
        for idx, row in page_df.iterrows():
            # Build images table
            # images_html = ''
            images_table = ''
            for key in image_keys:
                img = row.get(key, None)
                img_html = pil_to_html(img, max_width=max_width)
                # images_html += f'<td>{img_html}</td>'
                images_table += f'''
                <table class="image-table">
                    <tr><td>{img_html}</td></tr>
                </table>
                '''
            
            # Build texts table
            texts_html = ''
            for key in text_keys:
                text = row.get(key, '')
                # If text is a dict or DataFrame, render as a table
                if isinstance(text, pd.DataFrame):
                    text_html = text.to_html(index=False)
                elif isinstance(text, dict):
                    text_df = pd.DataFrame(list(text.items()), columns=['Key', key])
                    text_html = text_df.to_html(index=False)
                else:
                    text_html = str_to_html(text)
                texts_html += f'<td>{text_html}</td>'
            texts_table = f'''
            <table class="text-table">
                <tr>{texts_html}</tr>
            </table>
            '''
            
            # Combine the images and texts tables for the current item
            item_html = f'''
            <div class="item">
                {images_table}
                {texts_table}
            </div>
            '''
            page_html += item_html
        
        html_pages.append(f'<div class="page" id="page{page_num}" style="display:none;">{page_html}</div>')
    
    # Set the first page to display by default
    if html_pages:
        html_pages[0] = html_pages[0].replace('style="display:none;"', 'style="display:block;"')
    
    # Pagination controls
    pagination_controls = '''
    <div class="pagination">
        <button id="prevButton" onclick="prevPage()">Previous</button>
        <span id="pageNumber">Page 1</span>
        <button id="nextButton" onclick="nextPage()">Next</button>
    </div>
    '''
    
    # CSS styles
    css_styles = '''
    <style>
        .pagination {
            text-align: center;
            margin-bottom: 20px;
        }
        .item {
            margin-bottom: 40px;
        }
        .image-table, .text-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 10px;
        }
        .image-table td, .text-table th .text-table td {
            border: 1px solid #dddddd;
            text-align: center;
            vertical-align: top;
            padding: 5px;
            word-wrap: break-word;
        }
        .image-table img, .text-table img {
            max-width: 100%;
            height: auto;
        }
    </style>
    '''
    
    # JavaScript for pagination
    js_script = f'''
    <script>
        var currentPage = 0;
        var totalPages = {num_pages};

        function showPage(pageNum) {{
            for (var i = 0; i < totalPages; i++) {{
                document.getElementById('page' + i).style.display = 'none';
            }}
            document.getElementById('page' + pageNum).style.display = 'block';
            document.getElementById('pageNumber').innerText = 'Page ' + (pageNum + 1) + ' of ' + totalPages;
            document.getElementById('prevButton').disabled = pageNum === 0;
            document.getElementById('nextButton').disabled = pageNum === totalPages - 1;
            currentPage = pageNum;
        }}

        function prevPage() {{
            if (currentPage > 0) {{
                showPage(currentPage - 1);
            }}
        }}

        function nextPage() {{
            if (currentPage < totalPages - 1) {{
                showPage(currentPage + 1);
            }}
        }}

        document.addEventListener('DOMContentLoaded', function() {{
            showPage(0);
        }});
    </script>
    '''
    
    # Combine all parts
    final_html = f"""
    {css_styles}
    {pagination_controls}
    {''.join(html_pages)}
    {js_script}
    """
    return final_html

def convert_df_to_html_v2(df: pd.DataFrame, image_keys=None, text_keys=None, max_rows=300, max_width=300):
    if image_keys is None:
        image_keys = []
    if text_keys is None:
        text_keys = []
    
    html_pages = []
    total_rows = len(df)
    num_pages = (total_rows // max_rows) + (1 if total_rows % max_rows != 0 else 0)
    
    for page_num in range(num_pages):
        start_idx = page_num * max_rows
        end_idx = min(start_idx + max_rows, total_rows)
        page_df = df.iloc[start_idx:end_idx]
        page_html = ''
        
        for idx, row in page_df.iterrows():
            # Build images table
            images_html = ''
            for key in image_keys:
                img = row.get(key, None)
                img_html = pil_to_html(img, max_width=max_width)
                images_html += f'<td>{img_html}</td>'
            images_table = f'''
            <table class="image-table">
                <tr>{images_html}</tr>
            </table>
            '''
            
            # Build texts table with aligned rows
            # Collect data from text_keys
            text_data_frames = []
            max_num_rows = 0  # To find the maximum number of rows among text_keys
            
            for key in text_keys:
                text = row.get(key, '')
                if isinstance(text, pd.DataFrame):
                    df_text = text
                elif isinstance(text, dict):
                    text = {str(k): str(v) for k, v in text.items()}
                    df_text = pd.DataFrame(list(text.items()), columns=['Key', key])
                elif isinstance(text, list):
                    df_text = pd.DataFrame(text, columns=[key])
                else:
                    df_text = pd.DataFrame({key: [str_to_html(text)]})
                
                df_text = df_text.reset_index(drop=True)
                num_rows = len(df_text)
                if num_rows > max_num_rows:
                    max_num_rows = num_rows
                text_data_frames.append((key, df_text))
            
            # Align data by extending shorter data frames
            aligned_data = {}
            for key, df_text in text_data_frames:
                num_rows = len(df_text)
                if num_rows < max_num_rows:
                    # Extend df_text to have max_num_rows
                    additional_rows = pd.DataFrame([['']] * (max_num_rows - num_rows), columns=df_text.columns)
                    df_text = pd.concat([df_text, additional_rows], ignore_index=True)
                # Convert df_text to HTML strings per cell
                df_text = df_text.map(str_to_html)
                aligned_data[key] = df_text
            
            # Build the combined DataFrame
            combined_df = pd.DataFrame()
            for key, df_text in aligned_data.items():
                combined_df = pd.concat([combined_df, df_text], axis=1)
            
            # Now build the HTML table from combined_df
            texts_table_html = combined_df.to_html(index=False, escape=False)

            texts_table = f'''
            <table class="text-table">
                {texts_table_html}
            </table>
            '''
            
            # Combine the images and texts tables for the current item
            item_html = f'''
            <div class="item">
                {images_table}
                {texts_table}
            </div>
            '''
            page_html += item_html
        
        html_pages.append(f'<div class="page" id="page{page_num}" style="display:none;">{page_html}</div>')
    
        # Clear variables to free memory
        del page_df, page_html
    
    # Set the first page to display by default
    if html_pages:
        html_pages[0] = html_pages[0].replace('style="display:none;"', 'style="display:block;"')
    
    # Pagination controls
    pagination_controls = '''
    <div class="pagination">
        <button id="prevButton" onclick="prevPage()">Previous</button>
        <span id="pageNumber">Page 1</span>
        <button id="nextButton" onclick="nextPage()">Next</button>
    </div>
    '''
    
    # CSS styles
    css_styles = '''
    <style>
        .pagination {
            text-align: center;
            margin-bottom: 20px;
        }
        .item {
            margin-bottom: 40px;
        }
        .image-table, .text-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 10px;
        }
        .image-table td, .text-table th, .text-table td {
            border: 1px solid #dddddd;
            text-align: center;
            vertical-align: top;
            padding: 5px;
            word-wrap: break-word;
        }
        .image-table img, .text-table img {
            max-width: 100%;
            height: auto;
        }
        .text-table th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
    </style>
    '''
    
    # JavaScript for pagination
    js_script = f'''
    <script>
        var currentPage = 0;
        var totalPages = {num_pages};

        function showPage(pageNum) {{
            for (var i = 0; i < totalPages; i++) {{
                var page = document.getElementById('page' + i);
                if (page) {{
                    page.style.display = 'none';
                }}
            }}
            var currentPageDiv = document.getElementById('page' + pageNum);
            if (currentPageDiv) {{
                currentPageDiv.style.display = 'block';
            }}
            document.getElementById('pageNumber').innerText = 'Page ' + (pageNum + 1) + ' of ' + totalPages;
            document.getElementById('prevButton').disabled = pageNum === 0;
            document.getElementById('nextButton').disabled = pageNum === totalPages - 1;
            currentPage = pageNum;
        }}

        function prevPage() {{
            if (currentPage > 0) {{
                showPage(currentPage - 1);
            }}
        }}

        function nextPage() {{
            if (currentPage < totalPages - 1) {{
                showPage(currentPage + 1);
            }}
        }}

        document.addEventListener('DOMContentLoaded', function() {{
            showPage(0);
        }});
    </script>
    '''
    
    # Combine all parts
    final_html = f"""
    {css_styles}
    {pagination_controls}
    {''.join(html_pages)}
    {js_script}
    """
    return final_html