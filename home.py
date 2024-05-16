from st_pages import Page, show_pages, add_page_title
import streamlit as st

# Optional -- adds the title and icon to the current page
add_page_title()
show_pages(
    [
        Page("_pages/openai_pdf2kg.py", "OpenAI prompt pdf to KG", ":ice_cube:"),
        Page("_pages/openai_constructingKG.py", "OpenAI construct pdf to KG", ":building_construction:"),
        Page("_pages/openai_etl_hierarchy.py", "OpenAI build hierarchical pdf to KG", ":jigsaw:"),
        Page("_nasa/app.py", "Vector Only vs KG+Vector", ":rocket:"),       
    ]
)