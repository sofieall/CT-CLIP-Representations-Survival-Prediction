

def generate_notes_from_clinical_variables(patient_info):
    """
    Generates a clinical note from a patient's clinical variables.

    Args:
        patient_info (pd.Series): A pandas Series containing clinical
            information for a single patient.
    """

    # Note: The functions get_xxx_str need to be adapted to the dataset being used.
    sex = get_sex_str(patient_info)
    diagnosis = get_diagnosis_str(patient_info)
    stage = get_overall_stage_str(patient_info, diagnosis)
    t_stage = get_t_stage_str(patient_info)
    n_stage = get_n_stage_str(patient_info)
    tumor_location = get_location_tumor_str(patient_info)
    smoking_status = get_smoking_status_str(patient_info)
    performance_status = get_performance_status_str(patient_info)

    # Create clinical notes from tabular data
    if diagnosis and t_stage and smoking_status:
        clinical_prompt = f"{sex} {diagnosis} {stage}. {tumor_location} {t_stage}. {n_stage}. {smoking_status} {performance_status}"

    elif diagnosis and not t_stage and smoking_status: 
        clinical_prompt = f"{sex} {diagnosis} {stage}. {tumor_location} {n_stage}. {smoking_status} {performance_status}"

    elif diagnosis and t_stage and not smoking_status:
        clinical_prompt = f"{sex} {diagnosis} {stage}. {tumor_location} {t_stage}. {n_stage}. {performance_status}"

    elif not diagnosis and t_stage and smoking_status:
        clinical_prompt = f"{sex} {stage}. {tumor_location} {t_stage}. {n_stage}. {smoking_status} {performance_status}"

    elif not diagnosis and not t_stage and smoking_status: 
        clinical_prompt = f"{sex} {stage}. {tumor_location} {n_stage}. {smoking_status} {performance_status}"

    else: 
        raise Exception(f"The combination {diagnosis}, {t_stage}, {smoking_status} is not yet implemented")

    clinical_prompt = clinical_prompt.replace('"', '')
    clinical_prompt = clinical_prompt.replace('\'', '')
    clinical_prompt = clinical_prompt.replace('(', '')
    clinical_prompt = clinical_prompt.replace(')', '')

    return clinical_prompt


def get_sex_str(patient_info):
    sex_label = patient_info.kon_value
    if sex_label == 1:
        sex = f"Male lung cancer patient."
    elif sex_label == 2: 
        sex = f"Female lung cancer patient."
    else: 
        sex = None
    return sex


def get_diagnosis_str(patient_info):
    diagnosis_label = patient_info.a_diagnos
    if diagnosis_label == 1:
        diagnosis = f"The patient is diagnosed with squamous cell carcinoma,"
    elif diagnosis_label == 2:
        diagnosis = f"The patient is diagnosed with small cell lung cancer,"
    elif diagnosis_label == 3:
        diagnosis = f"The patient is diagnosed with adenocarcinoma,"
    elif diagnosis_label == 4:
        diagnosis = f"The patient is diagnosed with large cell lung cancer,"
    elif diagnosis_label == 5:
        diagnosis = f"The patient is diagnosed with adenosquamous carcinoma,"
    elif diagnosis_label == 7:
        diagnosis = f"The cancer is of type carcinoid,"
    elif diagnosis_label == 8:
        diagnosis = f"The patient is diagnosed with salivary gland-type lung cancer,"
    else: 
        diagnosis = None
    return diagnosis


def get_overall_stage_str(patient_info, diagnosis):
    stage_label = patient_info.d0_stadium
    if diagnosis:
        stage = f"overall stage {stage_label}"
    else: 
        stage = f"Overall cancer stage {stage_label}"
    return stage


def get_t_stage_str(patient_info):
    t_stage_label = patient_info.a_tumtny
    if t_stage_label == 1:
        t_number = "T1"
        t_explanation = "the cancer is contained within the lung"
    elif t_stage_label == 11:
        t_number = "T1a"
        t_explanation = "the cancer is 1 cm or less at its widest part"
    elif t_stage_label == 12:
        t_number = "T1b"
        t_explanation ="the cancer is between 1 cm and 2 cm across"
    elif t_stage_label == 13:
        t_number = "T1c"
        t_explanation = "the cancer is between 2 and 3 cm across"
    elif t_stage_label == 21:
        t_number = "T2a"
        t_explanation = "the cancer is between 3 cm and 4 cm"
    elif t_stage_label == 22:
        t_number = "T2b"
        t_explanation = "the cancer is between 4 cm and 5 cm"
    elif t_stage_label == 3:
        t_number = "T3"
        t_explanation = None 
    elif t_stage_label == 4:
        t_number = "T4"
        t_explanation = None 
    else:
        t_number = None
        t_explanation = None
    if t_number and t_explanation:
        t_stage = f"The tumor stage is {t_number}, which means {t_explanation}"
    elif t_number:
        t_stage = f"The tumor stage is {t_number}"
    else: 
        t_stage = None
    return t_stage


def get_n_stage_str(patient_info):
    n_stage_label = patient_info.a_tumn
    if n_stage_label == 0:
        n_number = "N0"
        n_explanation = "the lymph nodes do not contain cancer cells"
    elif n_stage_label == 1:
        n_number = "N1"
        n_explanation = "there are cancer cells in lymph nodes within the lung or the hilum"
    elif n_stage_label == 2:
        n_number = "N2"
        n_explanation = "there is cancer in lymph nodes"
    elif n_stage_label == 3:
        n_number = "N3"
        n_explanation = "there is cancer in lymph nodes"
    else:   
        n_number = None
        n_explanation = None
    if n_number and n_explanation:
        n_stage = f"The node stage is {n_number}, which means {n_explanation}"
    else: 
        n_stage = None
    return n_stage


def get_location_side_str(patient_info):
    location_side_label = patient_info.a_tumlagesid
    if location_side_label == 1:
        location_side = "right"  
    elif location_side_label == 2:
        location_side = "left"
    else: 
        raise Exception("Not allowed value of variable a_tumlagesid ") 
    return location_side


def get_location_tumor_str(patient_info):
    location_side = get_location_side_str(patient_info)
    location_lobe_label = patient_info.a_tumlagelob
    if  location_lobe_label == 1:
        location_lobe = "close to the main bronchus"
    elif location_lobe_label == 2:
        location_lobe = "upper lobe"
    elif location_lobe_label == 3:
        location_lobe = "middle lobe"
    elif location_lobe_label == 4:
        location_lobe = "lower lobe"
    elif location_lobe_label == 5:
        location_lobe = "and it has grown into nearby structures" 
    elif location_lobe_label == 6:
        location_lobe = "in multiple spots"
    else: 
        raise Exception("Not allowed value of variable a_tumlagesid ") 
    tumor_location = f"The cancer is in the {location_side} lung {location_lobe}."
    return tumor_location


def get_smoking_status_str(patient_info):
    smoking_status_label = patient_info.a_rokstatus
    if smoking_status_label == 1: 
        smoking_status = f"The patient is a smoker and has smoked daily for at least one year."
    elif smoking_status_label == 2: 
        smoking_status = f"The patient is a former smoker and has been smoke-free for more than one year."
    elif smoking_status_label == 3: 
        smoking_status = f"The patient is a non-smoker."
    else:
        smoking_status = None
    return smoking_status


def get_performance_status_str(patient_info):
    performance_status_label = patient_info.a_who
    if performance_status_label == 0: 
        performance_status = f"The ECOG performance status of the patient is 0, which means the patient is fully active and able to carry on all pre-disease performance without restriction."
    elif performance_status_label == 1: 
        performance_status = f"The ECOG performance status of the patient is 1, which means the patient is restricted in physically challenging activity but ambulatory and able to carry out work of a light or sedentary nature, e.g., office work."
    elif performance_status_label == 2: 
        performance_status = f"The ECOG performance status of the patient is 2, which means the patient is ambulatory and capable of all selfcare but unable to carry out any work activities. The patient is up and about more than 50% of waking hours."
    elif performance_status_label == 3: 
        performance_status = f"The ECOG performance status of the patient is 3, which means e patient is capable of only limited selfcare and is confined to bed or chair more than 50% of waking hours."
    elif performance_status_label == 4: 
        performance_status = f"The ECOG performance status of the patient is 4, which means the patient is completely disabled and cannot carry on any selfcare. The patient is totally confined to bed or chair."
    else: 
        performance_status = None
    return performance_status


