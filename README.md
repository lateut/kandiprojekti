# Kandiprojekti
Posture Checker – Käyttöohje 

1. Sovelluksen tarkoitus 
Posture Checker on ergonomian seurantaan tarkoitettu sovellus, joka käyttää tietokoneen kameraa käyttäjän työasennon analysointiin. Sovellus tunnistaa käyttäjän ylävartalon asennon MediaPipe Pose Landmarker -mallin avulla ja antaa palautetta, jos työasento vaikuttaa huonolta. 
Sovellus toimii paikallisesti käyttäjän omalla tietokoneella, eikä videokuvaa lähetetä ulkoisiin palveluihin. 
 

2. Tarvittavat ohjelmistot ja tiedostot 

Sovelluksen käyttöön tarvitaan: 
- Python  
- OpenCV  
- MediaPipe  
- NumPy  
- Web-kamera  

MediaPipe-mallitiedosto: 
- pose_landmarker_lite.task

Mallitiedoston täytyy olla samassa kansiossa sovelluksen Python-tiedoston kanssa. 
Jos mallitiedostoa ei löydy, sovellus ei käynnisty ja näyttää virheilmoituksen. 


3. Sovelluksen käynnistäminen 
Avaa komentorivi tai terminaali sovelluksen kansiossa ja suorita ohjelma komennolla:
- python posture_checker_v8.py 

Jos tiedoston nimi on eri, käytä oman Python-tiedostosi nimeä. 
Käynnistyksen jälkeen avautuu aloitusvalikko.
