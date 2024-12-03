import streamlit as st
import fitz  # PyMuPDF
import spacy
import io
import base64
import json
from PIL import Image
import numpy as np

class ArtworkPDFExtractor:
    def __init__(self, pdf_file):
        """
        Initialise l'extracteur avec un fichier PDF en mémoire
        
        Args:
            pdf_file (bytes): Contenu du fichier PDF
        """
        # Convertir le fichier en document PyMuPDF
        self.document = fitz.open(stream=pdf_file, filetype="pdf")
        
        # Charger le modèle NLP français
        try:
            self.nlp = spacy.load('fr_core_news_md')
        except OSError:
            st.warning("Modèle spaCy non trouvé. Certaines fonctionnalités seront limitées.")
            self.nlp = None

    def extract_images(self):
        """
        Extrait toutes les images du PDF
        
        Returns:
            list: Images extraites (format PIL)
        """
        images = []
        for page_num in range(len(self.document)):
            page = self.document[page_num]
            
            # Extraction des images
            for img in page.get_images(full=True):
                xref = img[0]
                base_image = self.document.extract_image(xref)
                image_bytes = base_image['image']
                
                # Convertir en image PIL
                pil_image = Image.open(io.BytesIO(image_bytes))
                images.append(pil_image)
        
        return images

    def extract_text(self):
        """
        Extrait le texte de tous les PDF
        
        Returns:
            str: Texte complet du document
        """
        full_text = ""
        for page in self.document:
            full_text += page.get_text()
        return full_text

    def identify_artwork_info(self, text):
        """
        Identifie les informations sur les œuvres d'art
        
        Args:
            text (str): Texte à analyser
        
        Returns:
            dict: Informations extraites
        """
        if not self.nlp:
            return {"erreur": "Modèle NLP non disponible"}

        doc = self.nlp(text)
        
        # Extraction d'informations potentiellement pertinentes
        artwork_info = {
            'artistes': set(),
            'titres': set(),
            'dates': set(),
            'techniques': set(),
            'dimensions': set()
        }
        
        # Extraction des entités nommées
        for ent in doc.ents:
            if ent.label_ == 'PER':
                artwork_info['artistes'].add(ent.text)
            elif ent.label_ == 'WORK_OF_ART':
                artwork_info['titres'].add(ent.text)
        
        # Recherche de motifs textuels
        for sent in doc.sents:
            # Exemple de motifs à détecter (à affiner)
            if any(mot in sent.text.lower() for mot in ['date', 'année', 'créé en']):
                artwork_info['dates'].add(sent.text)
            if any(mot in sent.text.lower() for mot in ['technique', 'huile', 'aquarelle', 'acrylique']):
                artwork_info['techniques'].add(sent.text)
        
        return artwork_info

def main():
    st.title("Extracteur d'Œuvres d'Art à partir de PDF")
    
    # Upload du fichier PDF
    uploaded_file = st.file_uploader("Télécharger un fichier PDF", type="pdf")
    
    if uploaded_file is not None:
        # Lire le contenu du fichier
        pdf_bytes = uploaded_file.getvalue()
        
        # Créer l'extracteur
        extractor = ArtworkPDFExtractor(pdf_bytes)
        
        # Sections d'affichage
        st.header("Résultats de l'extraction")
        
        # Extraction du texte
        st.subheader("Texte Extrait")
        full_text = extractor.extract_text()
        st.text_area("Contenu du PDF", full_text, height=200)
        
        # Extraction des informations sur les œuvres
        st.subheader("Informations sur les Œuvres")
        artwork_info = extractor.identify_artwork_info(full_text)
        st.json(artwork_info)
        
        # Extraction et affichage des images
        st.subheader("Images Extraites")
        images = extractor.extract_images()
        
        # Afficher les images dans des colonnes
        cols = st.columns(min(3, len(images) + 1))
        for i, img in enumerate(images):
            with cols[i % 3]:
                st.image(img, caption=f"Image {i+1}", use_column_width=True)

if __name__ == '__main__':
    main()

# Dépendances requises :
# pip install streamlit pymupdf spacy pillow
# python -m spacy download fr_core_news_md
