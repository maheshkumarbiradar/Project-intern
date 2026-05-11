"""
doctors.py - Doctor recommendation logic based on tumor type
Covers all major Indian cities with top hospitals
"""

# ── Tumor → Specialist mapping ────────────────────────────────────────────────
TUMOR_INFO = {
    "glioma": {
        "specialist":  "Neuro-Oncologist",
        "description": "Gliomas are tumors that arise from glial cells in the brain or spine. "
                       "They range from slow-growing (Grade I) to aggressive (Grade IV/GBM).",
        "urgency":     "High — schedule consultation within 1–2 weeks.",
        "departments": ["Neuro-Oncology", "Neurosurgery", "Radiation Oncology"],
    },
    "meningioma": {
        "specialist":  "Neurosurgeon",
        "description": "Meningiomas arise from the meninges (brain/spinal cord covering). "
                       "Most are benign and slow-growing; treatment depends on size and symptoms.",
        "urgency":     "Moderate — consult within 2–4 weeks.",
        "departments": ["Neurosurgery", "Radiology", "Neurology"],
    },
    "pituitary": {
        "specialist":  "Endocrinologist / Neurosurgeon",
        "description": "Pituitary tumors affect the pituitary gland and can disrupt hormone "
                       "production. Most are benign adenomas.",
        "urgency":     "Moderate — consult within 2–4 weeks.",
        "departments": ["Endocrinology", "Neurosurgery", "Ophthalmology"],
    },
    "notumor": {
        "specialist":  "General Neurologist",
        "description": "No tumor detected. However, if symptoms persist, further evaluation "
                       "by a neurologist is recommended.",
        "urgency":     "Low — routine follow-up if symptoms are present.",
        "departments": ["Neurology", "General Medicine"],
    },
}

# ── Hospital Database — All Major Indian Cities ───────────────────────────────
HOSPITAL_DATABASE = [

    # ── CHENNAI ──────────────────────────────────────────────────────────────
    {
        "name": "Apollo Hospitals", "city": "Chennai",
        "address": "21 Greams Lane, Off Greams Road, Chennai - 600006",
        "phone": "+91 44 2829 3333", "website": "https://www.apollohospitals.com",
        "rating": 4.7,
        "departments": ["Neuro-Oncology", "Neurosurgery", "Endocrinology", "Radiation Oncology", "Neurology"],
    },
    {
        "name": "Fortis Malar Hospital", "city": "Chennai",
        "address": "52 1st Main Road, Gandhi Nagar, Adyar, Chennai - 600020",
        "phone": "+91 44 4289 2222", "website": "https://www.fortishealthcare.com",
        "rating": 4.5,
        "departments": ["Neurosurgery", "Neurology", "Radiation Oncology", "Endocrinology"],
    },
    {
        "name": "MIOT International", "city": "Chennai",
        "address": "4/112 Mount Poonamallee Road, Manapakkam, Chennai - 600089",
        "phone": "+91 44 4200 2288", "website": "https://www.miothospitals.com",
        "rating": 4.6,
        "departments": ["Neurosurgery", "Neuro-Oncology", "Neurology", "Endocrinology"],
    },
    {
        "name": "Sri Ramachandra Medical Centre", "city": "Chennai",
        "address": "1 Ramachandra Nagar, Porur, Chennai - 600116",
        "phone": "+91 44 4592 8600", "website": "https://www.sriramachandra.edu.in",
        "rating": 4.5,
        "departments": ["Neurology", "Neurosurgery", "Oncology", "Endocrinology", "General Medicine"],
    },
    {
        "name": "Kauvery Hospital", "city": "Chennai",
        "address": "199 Luz Church Road, Mylapore, Chennai - 600004",
        "phone": "+91 44 4000 6000", "website": "https://www.kauveryhospital.com",
        "rating": 4.4,
        "departments": ["Neurology", "Neurosurgery", "General Medicine"],
    },

    # ── MUMBAI ───────────────────────────────────────────────────────────────
    {
        "name": "Tata Memorial Hospital", "city": "Mumbai",
        "address": "Dr Ernest Borges Road, Parel, Mumbai - 400012",
        "phone": "+91 22 2417 7000", "website": "https://tmc.gov.in",
        "rating": 4.8,
        "departments": ["Neuro-Oncology", "Radiation Oncology", "Neurosurgery", "Oncology"],
    },
    {
        "name": "Kokilaben Dhirubhai Ambani Hospital", "city": "Mumbai",
        "address": "Rao Saheb Achutrao Patwardhan Marg, Four Bungalows, Andheri West, Mumbai - 400053",
        "phone": "+91 22 4269 6969", "website": "https://www.kokilabenhospital.com",
        "rating": 4.7,
        "departments": ["Neurosurgery", "Neuro-Oncology", "Neurology", "Endocrinology", "Radiation Oncology"],
    },
    {
        "name": "Lilavati Hospital", "city": "Mumbai",
        "address": "A-791, Bandra Reclamation, Bandra West, Mumbai - 400050",
        "phone": "+91 22 2675 1000", "website": "https://www.lilavatihospital.com",
        "rating": 4.6,
        "departments": ["Neurology", "Neurosurgery", "Endocrinology", "General Medicine"],
    },
    {
        "name": "Hinduja Hospital", "city": "Mumbai",
        "address": "Veer Savarkar Marg, Mahim, Mumbai - 400016",
        "phone": "+91 22 2445 1515", "website": "https://www.hindujahospital.com",
        "rating": 4.5,
        "departments": ["Neurosurgery", "Neurology", "Radiation Oncology", "Endocrinology"],
    },
    {
        "name": "Bombay Hospital", "city": "Mumbai",
        "address": "12 New Marine Lines, Mumbai - 400020",
        "phone": "+91 22 2206 7676", "website": "https://www.bombayhospital.com",
        "rating": 4.4,
        "departments": ["Neurology", "Neurosurgery", "General Medicine", "Endocrinology"],
    },

    # ── DELHI ────────────────────────────────────────────────────────────────
    {
        "name": "AIIMS Delhi", "city": "Delhi",
        "address": "Sri Aurobindo Marg, Ansari Nagar, New Delhi - 110029",
        "phone": "+91 11 2659 3308", "website": "https://www.aiims.edu",
        "rating": 4.9,
        "departments": ["Neuro-Oncology", "Neurosurgery", "Neurology", "Endocrinology", "Radiation Oncology"],
    },
    {
        "name": "Fortis Memorial Research Institute", "city": "Delhi",
        "address": "Sector 44, Opposite HUDA City Centre, Gurugram - 122002",
        "phone": "+91 124 4962 200", "website": "https://www.fortishealthcare.com",
        "rating": 4.7,
        "departments": ["Neurosurgery", "Neuro-Oncology", "Neurology", "Endocrinology"],
    },
    {
        "name": "Max Super Speciality Hospital", "city": "Delhi",
        "address": "1 Press Enclave Road, Saket, New Delhi - 110017",
        "phone": "+91 11 2651 5050", "website": "https://www.maxhealthcare.in",
        "rating": 4.6,
        "departments": ["Neurosurgery", "Neurology", "Radiation Oncology", "Endocrinology"],
    },
    {
        "name": "Medanta The Medicity", "city": "Delhi",
        "address": "CH Baktawar Singh Road, Sector 38, Gurugram - 122001",
        "phone": "+91 124 4141 414", "website": "https://www.medanta.org",
        "rating": 4.7,
        "departments": ["Neuro-Oncology", "Neurosurgery", "Endocrinology", "Neurology"],
    },
    {
        "name": "Sir Ganga Ram Hospital", "city": "Delhi",
        "address": "Rajinder Nagar, New Delhi - 110060",
        "phone": "+91 11 2575 7575", "website": "https://www.sgrh.com",
        "rating": 4.5,
        "departments": ["Neurology", "Neurosurgery", "Endocrinology", "General Medicine"],
    },

    # ── BENGALURU ────────────────────────────────────────────────────────────
    {
        "name": "Manipal Hospitals", "city": "Bengaluru",
        "address": "98 HAL Airport Road, Bengaluru - 560017",
        "phone": "+91 80 2502 4444", "website": "https://www.manipalhospitals.com",
        "rating": 4.7,
        "departments": ["Neuro-Oncology", "Neurosurgery", "Neurology", "Endocrinology", "Radiation Oncology"],
    },
    {
        "name": "Narayana Health City", "city": "Bengaluru",
        "address": "258/A Bommasandra Industrial Area, Anekal Taluk, Bengaluru - 560099",
        "phone": "+91 80 7122 2222", "website": "https://www.narayanahealth.org",
        "rating": 4.6,
        "departments": ["Neurosurgery", "Neuro-Oncology", "Neurology", "Endocrinology"],
    },
    {
        "name": "Fortis Hospital Bangalore", "city": "Bengaluru",
        "address": "14 Cunningham Road, Bengaluru - 560052",
        "phone": "+91 80 6621 4444", "website": "https://www.fortishealthcare.com",
        "rating": 4.5,
        "departments": ["Neurology", "Neurosurgery", "Radiation Oncology", "Endocrinology"],
    },
    {
        "name": "Apollo Hospital Bangalore", "city": "Bengaluru",
        "address": "154/11 Opp IIM-B, Bannerghatta Road, Bengaluru - 560076",
        "phone": "+91 80 2630 4050", "website": "https://www.apollohospitals.com",
        "rating": 4.6,
        "departments": ["Neuro-Oncology", "Neurosurgery", "Endocrinology", "General Medicine"],
    },
    {
        "name": "NIMHANS", "city": "Bengaluru",
        "address": "Hosur Road, Bengaluru - 560029",
        "phone": "+91 80 4611 5555", "website": "https://nimhans.ac.in",
        "rating": 4.8,
        "departments": ["Neurology", "Neurosurgery", "General Medicine"],
    },

    # ── HYDERABAD ────────────────────────────────────────────────────────────
    {
        "name": "KIMS Hospital", "city": "Hyderabad",
        "address": "1-8-31/1 Krishna Institute of Medical Sciences, Minister Road, Secunderabad - 500003",
        "phone": "+91 40 4488 5000", "website": "https://www.kimshospitals.com",
        "rating": 4.7,
        "departments": ["Neurosurgery", "Neuro-Oncology", "Neurology", "Endocrinology"],
    },
    {
        "name": "Apollo Hospital Hyderabad", "city": "Hyderabad",
        "address": "Jubilee Hills, Hyderabad - 500033",
        "phone": "+91 40 2360 7777", "website": "https://www.apollohospitals.com",
        "rating": 4.6,
        "departments": ["Neuro-Oncology", "Neurosurgery", "Endocrinology", "Radiation Oncology"],
    },
    {
        "name": "Yashoda Hospitals", "city": "Hyderabad",
        "address": "Raj Bhavan Road, Somajiguda, Hyderabad - 500082",
        "phone": "+91 40 4567 4567", "website": "https://www.yashodahospitals.com",
        "rating": 4.5,
        "departments": ["Neurology", "Neurosurgery", "Endocrinology", "General Medicine"],
    },
    {
        "name": "Care Hospitals", "city": "Hyderabad",
        "address": "Road No 1, Banjara Hills, Hyderabad - 500034",
        "phone": "+91 40 3041 8888", "website": "https://www.carehospitals.com",
        "rating": 4.4,
        "departments": ["Neurosurgery", "Neurology", "Radiation Oncology"],
    },
    {
        "name": "Nizam's Institute of Medical Sciences", "city": "Hyderabad",
        "address": "Panjagutta, Hyderabad - 500082",
        "phone": "+91 40 2348 9000", "website": "https://nims.ap.nic.in",
        "rating": 4.5,
        "departments": ["Neurology", "Neurosurgery", "Endocrinology", "General Medicine"],
    },

    # ── KOLKATA ──────────────────────────────────────────────────────────────
    {
        "name": "Apollo Gleneagles Hospital", "city": "Kolkata",
        "address": "58 Canal Circular Road, Kolkata - 700054",
        "phone": "+91 33 2320 3040", "website": "https://www.apollohospitals.com",
        "rating": 4.6,
        "departments": ["Neuro-Oncology", "Neurosurgery", "Neurology", "Endocrinology"],
    },
    {
        "name": "Fortis Hospital Kolkata", "city": "Kolkata",
        "address": "730 Anandapur, EM Bypass, Kolkata - 700107",
        "phone": "+91 33 6628 4444", "website": "https://www.fortishealthcare.com",
        "rating": 4.5,
        "departments": ["Neurosurgery", "Neurology", "Radiation Oncology", "Endocrinology"],
    },
    {
        "name": "AMRI Hospital", "city": "Kolkata",
        "address": "P-4 & 5 CIT Scheme LXXII, Gariahat Road South, Dhakuria, Kolkata - 700029",
        "phone": "+91 33 6606 3800", "website": "https://www.amrihospitals.in",
        "rating": 4.4,
        "departments": ["Neurology", "Neurosurgery", "General Medicine", "Endocrinology"],
    },
    {
        "name": "Peerless Hospital", "city": "Kolkata",
        "address": "360 Panchasayar, Kolkata - 700094",
        "phone": "+91 33 4011 1222", "website": "https://www.peerlesshospital.com",
        "rating": 4.3,
        "departments": ["Neurology", "General Medicine", "Endocrinology"],
    },

    # ── PUNE ─────────────────────────────────────────────────────────────────
    {
        "name": "Ruby Hall Clinic", "city": "Pune",
        "address": "40 Sassoon Road, Pune - 411001",
        "phone": "+91 20 6645 5100", "website": "https://www.rubyhall.com",
        "rating": 4.6,
        "departments": ["Neurosurgery", "Neurology", "Endocrinology", "Radiation Oncology"],
    },
    {
        "name": "Jehangir Hospital", "city": "Pune",
        "address": "32 Sassoon Road, Pune - 411001",
        "phone": "+91 20 6681 5000", "website": "https://www.jehangir-hospital.com",
        "rating": 4.5,
        "departments": ["Neurology", "Neurosurgery", "General Medicine", "Endocrinology"],
    },
    {
        "name": "Sahyadri Hospital", "city": "Pune",
        "address": "30C Karve Road, Deccan Gymkhana, Pune - 411004",
        "phone": "+91 20 6721 5000", "website": "https://www.sahyadrihospital.com",
        "rating": 4.4,
        "departments": ["Neurosurgery", "Neurology", "Endocrinology", "General Medicine"],
    },
    {
        "name": "KEM Hospital Pune", "city": "Pune",
        "address": "489 Rasta Peth, Sardar Moodliar Road, Pune - 411011",
        "phone": "+91 20 6123 0000", "website": "https://www.kemhospital.org",
        "rating": 4.3,
        "departments": ["Neurology", "General Medicine", "Endocrinology"],
    },

    # ── AHMEDABAD ────────────────────────────────────────────────────────────
    {
        "name": "Apollo Hospital Ahmedabad", "city": "Ahmedabad",
        "address": "Plot No 1A, Bhat GIDC Estate, Gandhinagar - 382428",
        "phone": "+91 79 6670 1800", "website": "https://www.apollohospitals.com",
        "rating": 4.6,
        "departments": ["Neuro-Oncology", "Neurosurgery", "Endocrinology", "Neurology"],
    },
    {
        "name": "Sterling Hospital", "city": "Ahmedabad",
        "address": "Sterling Hospital Road, Gurukul, Ahmedabad - 380052",
        "phone": "+91 79 4000 4000", "website": "https://www.sterlinghospitals.com",
        "rating": 4.5,
        "departments": ["Neurology", "Neurosurgery", "Endocrinology", "General Medicine"],
    },
    {
        "name": "HCG Cancer Centre", "city": "Ahmedabad",
        "address": "Commerce Six Road, Navrangpura, Ahmedabad - 380009",
        "phone": "+91 79 3061 0000", "website": "https://www.hcgoncology.com",
        "rating": 4.5,
        "departments": ["Neuro-Oncology", "Radiation Oncology", "Oncology"],
    },

    # ── JAIPUR ───────────────────────────────────────────────────────────────
    {
        "name": "Fortis Escorts Hospital Jaipur", "city": "Jaipur",
        "address": "Jawaharlal Nehru Marg, Malviya Nagar, Jaipur - 302017",
        "phone": "+91 141 2547 000", "website": "https://www.fortishealthcare.com",
        "rating": 4.5,
        "departments": ["Neurosurgery", "Neurology", "Endocrinology", "Radiation Oncology"],
    },
    {
        "name": "Narayana Multispecialty Hospital Jaipur", "city": "Jaipur",
        "address": "Sector 28, Pratap Nagar, Jaipur - 302033",
        "phone": "+91 141 7116 000", "website": "https://www.narayanahealth.org",
        "rating": 4.4,
        "departments": ["Neurology", "Neurosurgery", "Endocrinology", "General Medicine"],
    },
    {
        "name": "SMS Hospital Jaipur", "city": "Jaipur",
        "address": "JLN Marg, Jaipur - 302004",
        "phone": "+91 141 2518 888", "website": "https://www.smshospitaljaipur.com",
        "rating": 4.3,
        "departments": ["Neurology", "Neurosurgery", "General Medicine"],
    },

    # ── LUCKNOW ──────────────────────────────────────────────────────────────
    {
        "name": "SGPGI Lucknow", "city": "Lucknow",
        "address": "Raebareli Road, Lucknow - 226014",
        "phone": "+91 522 2668 700", "website": "https://www.sgpgi.ac.in",
        "rating": 4.7,
        "departments": ["Neurology", "Neurosurgery", "Endocrinology", "Neuro-Oncology"],
    },
    {
        "name": "Medanta Hospital Lucknow", "city": "Lucknow",
        "address": "Sector A, Pocket 1, Amar Shaheed Path, Sushant Golf City, Lucknow - 226030",
        "phone": "+91 522 4505 050", "website": "https://www.medanta.org",
        "rating": 4.5,
        "departments": ["Neurosurgery", "Neurology", "Endocrinology", "Radiation Oncology"],
    },

    # ── CHANDIGARH ───────────────────────────────────────────────────────────
    {
        "name": "PGIMER Chandigarh", "city": "Chandigarh",
        "address": "Sector 12, Chandigarh - 160012",
        "phone": "+91 172 2756 565", "website": "https://pgimer.edu.in",
        "rating": 4.8,
        "departments": ["Neurology", "Neurosurgery", "Endocrinology", "Neuro-Oncology", "Radiation Oncology"],
    },
    {
        "name": "Fortis Hospital Mohali", "city": "Chandigarh",
        "address": "Sector 62, Phase VIII, Mohali - 160062",
        "phone": "+91 172 4692 222", "website": "https://www.fortishealthcare.com",
        "rating": 4.5,
        "departments": ["Neurosurgery", "Neurology", "Endocrinology"],
    },

    # ── KOCHI ────────────────────────────────────────────────────────────────
    {
        "name": "Amrita Institute of Medical Sciences", "city": "Kochi",
        "address": "AIMS Ponekkara PO, Kochi - 682041",
        "phone": "+91 484 2801 234", "website": "https://www.aims.amrita.edu",
        "rating": 4.8,
        "departments": ["Neuro-Oncology", "Neurosurgery", "Neurology", "Endocrinology", "Radiation Oncology"],
    },
    {
        "name": "Lakeshore Hospital", "city": "Kochi",
        "address": "Maradu, NH-47 By Pass, Kochi - 682304",
        "phone": "+91 484 2701 032", "website": "https://www.lakeshorehospital.com",
        "rating": 4.5,
        "departments": ["Neurology", "Neurosurgery", "Endocrinology", "General Medicine"],
    },

    # ── NAGPUR ───────────────────────────────────────────────────────────────
    {
        "name": "Wockhardt Hospital Nagpur", "city": "Nagpur",
        "address": "Trimurti Nagar, Nagpur - 440022",
        "phone": "+91 712 6188 100", "website": "https://www.wockhardthospitals.com",
        "rating": 4.4,
        "departments": ["Neurology", "Neurosurgery", "Endocrinology", "General Medicine"],
    },
    {
        "name": "Care Hospital Nagpur", "city": "Nagpur",
        "address": "Plot No 107, Opp Vivekanand Nagar Bus Stop, Nagpur - 440015",
        "phone": "+91 712 6190 000", "website": "https://www.carehospitals.com",
        "rating": 4.3,
        "departments": ["Neurosurgery", "Neurology", "General Medicine"],
    },

    # ── BHOPAL ───────────────────────────────────────────────────────────────
    {
        "name": "AIIMS Bhopal", "city": "Bhopal",
        "address": "Saket Nagar, Bhopal - 462020",
        "phone": "+91 755 2672 355", "website": "https://www.aiimsbhopal.edu.in",
        "rating": 4.7,
        "departments": ["Neurology", "Neurosurgery", "Endocrinology", "General Medicine"],
    },
    {
        "name": "Bansal Hospital Bhopal", "city": "Bhopal",
        "address": "C Sector, Shahpura, Bhopal - 462016",
        "phone": "+91 755 4000 000", "website": "https://www.bansalhospital.com",
        "rating": 4.3,
        "departments": ["Neurology", "Neurosurgery", "General Medicine"],
    },

    # ── VISAKHAPATNAM ────────────────────────────────────────────────────────
    {
        "name": "Seven Hills Hospital", "city": "Visakhapatnam",
        "address": "Rockdale Layout, Visakhapatnam - 530002",
        "phone": "+91 891 2564 444", "website": "https://www.7hillshospital.com",
        "rating": 4.4,
        "departments": ["Neurology", "Neurosurgery", "Endocrinology", "General Medicine"],
    },
    {
        "name": "KIMS Vizag", "city": "Visakhapatnam",
        "address": "Near Old Town, Asilmetta, Visakhapatnam - 530003",
        "phone": "+91 891 2588 888", "website": "https://www.kimshospitals.com",
        "rating": 4.3,
        "departments": ["Neurosurgery", "Neurology", "Endocrinology"],
    },

    # ── COIMBATORE ───────────────────────────────────────────────────────────
    {
        "name": "PSG Hospitals", "city": "Coimbatore",
        "address": "Peelamedu, Coimbatore - 641004",
        "phone": "+91 422 4345 000", "website": "https://www.psghospitals.com",
        "rating": 4.5,
        "departments": ["Neurology", "Neurosurgery", "Endocrinology", "General Medicine"],
    },
    {
        "name": "Kovai Medical Center", "city": "Coimbatore",
        "address": "99 Avanashi Road, Coimbatore - 641014",
        "phone": "+91 422 4323 800", "website": "https://www.kmchhospital.com",
        "rating": 4.4,
        "departments": ["Neurosurgery", "Neurology", "Endocrinology"],
    },

    # ── INDORE ───────────────────────────────────────────────────────────────
    {
        "name": "Bombay Hospital Indore", "city": "Indore",
        "address": "Ring Road, Indore - 452010",
        "phone": "+91 731 4077 000", "website": "https://www.bombayhospitalindore.com",
        "rating": 4.4,
        "departments": ["Neurology", "Neurosurgery", "Endocrinology", "General Medicine"],
    },
    {
        "name": "Medanta Hospital Indore", "city": "Indore",
        "address": "Sector D, Scheme 114, Vijay Nagar, Indore - 452010",
        "phone": "+91 731 4747 000", "website": "https://www.medanta.org",
        "rating": 4.5,
        "departments": ["Neurosurgery", "Neuro-Oncology", "Neurology", "Endocrinology"],
    },

    # ── PATNA ────────────────────────────────────────────────────────────────
    {
        "name": "Paras HMRI Hospital", "city": "Patna",
        "address": "Raja Bazar, Patna - 800014",
        "phone": "+91 612 3540 000", "website": "https://www.parashospitals.com",
        "rating": 4.3,
        "departments": ["Neurology", "Neurosurgery", "General Medicine", "Endocrinology"],
    },
    {
        "name": "AIIMS Patna", "city": "Patna",
        "address": "Phulwari Sharif, Patna - 801507",
        "phone": "+91 612 2451 070", "website": "https://www.aiimspatna.org",
        "rating": 4.6,
        "departments": ["Neurology", "Neurosurgery", "Endocrinology", "General Medicine"],
    },

    # ── SURAT ────────────────────────────────────────────────────────────────
    {
        "name": "Kiran Hospital Surat", "city": "Surat",
        "address": "Prashant Nagar, Katargam, Surat - 395004",
        "phone": "+91 261 2222 222", "website": "https://www.kiranhospital.com",
        "rating": 4.4,
        "departments": ["Neurology", "Neurosurgery", "Endocrinology", "General Medicine"],
    },
    {
        "name": "Sunshine Global Hospital", "city": "Surat",
        "address": "Dumas Road, Surat - 395007",
        "phone": "+91 261 6691 000", "website": "https://www.sunshinehospital.org",
        "rating": 4.3,
        "departments": ["Neurosurgery", "Neurology", "General Medicine"],
    },
]


def get_tumor_info(predicted_class: str) -> dict:
    """Return specialist info for a given tumor class."""
    return TUMOR_INFO.get(predicted_class, TUMOR_INFO["notumor"])


def get_available_cities() -> list:
    """Return sorted list of all cities in the database."""
    return sorted(set(h["city"] for h in HOSPITAL_DATABASE))


def recommend_doctors(predicted_class: str, city: str = "Chennai", top_n: int = 3) -> list:
    """
    Return top N hospitals for the tumor type in the given city.
    Falls back to nearby hospitals if city not found.
    """
    info         = get_tumor_info(predicted_class)
    needed_depts = set(info["departments"])
    city_lower   = city.lower()

    matches = [
        h for h in HOSPITAL_DATABASE
        if h["city"].lower() == city_lower
        and needed_depts & set(h["departments"])
    ]

    # Fallback: if no hospitals found in city, return top hospitals overall
    if not matches:
        matches = [
            h for h in HOSPITAL_DATABASE
            if needed_depts & set(h["departments"])
        ]

    matches.sort(key=lambda h: h["rating"], reverse=True)
    return matches[:top_n]


def format_recommendation(predicted_class: str, city: str = "Chennai") -> str:
    """Return a human-readable recommendation string."""
    info      = get_tumor_info(predicted_class)
    hospitals = recommend_doctors(predicted_class, city)

    lines = [
        f"\n{'='*55}",
        f"  MEDICAL RECOMMENDATION",
        f"{'='*55}",
        f"  Condition   : {predicted_class.upper()}",
        f"  Specialist  : {info['specialist']}",
        f"  Urgency     : {info['urgency']}",
        f"\n  About       : {info['description']}",
        f"\n  Relevant Departments: {', '.join(info['departments'])}",
        f"\n{'-'*55}",
        f"  Recommended Hospitals in {city}:",
        f"{'-'*55}",
    ]

    for i, h in enumerate(hospitals, 1):
        lines += [
            f"\n  {i}. {h['name']}  ⭐ {h['rating']}",
            f"     {h['address']}",
            f"     📞 {h['phone']}",
            f"     🌐 {h['website']}",
        ]

    lines.append(f"\n{'='*55}")
    lines.append("  ⚠️  Always consult a qualified medical professional.")
    lines.append(f"{'='*55}\n")
    return "\n".join(lines)


if __name__ == "__main__":
    print(f"Total hospitals : {len(HOSPITAL_DATABASE)}")
    print(f"Cities covered  : {len(get_available_cities())}")
    print(f"Cities          : {', '.join(get_available_cities())}")
