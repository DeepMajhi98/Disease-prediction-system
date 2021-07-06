from flask import Flask, render_template, request, flash, redirect
import pickle
import pandas as pd
import numpy as np





app = Flask(__name__)


# -----------Models--------------
modelH = pickle.load(open('Model/heart.pkl','rb'))
modelB = pickle.load(open('Model/cancer.pkl','rb'))
modelL = pickle.load(open('Model/liver.pkl','rb'))
modelD = pickle.load(open('Model/diabetes_new.pkl','rb'))
modelK = pickle.load(open('Model/kidney.pkl','rb'))















# --------------------Url Routing-------------------

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('Diabetes_form.html')

@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('Breast_Cancer_Form.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('Heart_Form.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('Kidney_form.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('Liver_Form.html')






# --------Route for Visualization button in home page--------------------
# @app.route("/visualizationHome", methods=['GET', 'POST'])
# def visualHome():
# 	return render_template('visualization.html')



#------------Route for Heart visualization page----------------------
# @app.route("/visualizationHeart", methods=['GET', 'POST'])
# def visualHeart():
# 	return render_template('Heart_viz_page.html')



#------------Route for Breast Cancer visualization page----------------------
# @app.route("/visualizationBC", methods=['GET', 'POST'])
# def visualBC():
# 	return render_template('BC_viz_page.html')




#------------Route for Diabetes visualization page----------------------
# @app.route("/visualizationDiabetes", methods=['GET', 'POST'])
# def visualDiabetes():
# 	return render_template('Diabetes_viz_page.html')




#------------Route for Kidney visualization page----------------------
# @app.route("/visualizationKidney", methods=['GET', 'POST'])
# def visualKidney():
# 	return render_template('Kidney_viz_page.html')




#------------Route for Liver visualization page----------------------
# @app.route("/visualizationLiver", methods=['GET', 'POST'])
# def visualLiver():
# 	return render_template('Liver_viz_page.html')















# -------------Prediction for Heart----------------

@app.route('/predictH',methods=['POST'])
def predictHeart():
    input_featuresH = [float(x) for x in request.form.values()]
    features_valueH = [np.array(input_featuresH)]
    
    features_nameH = ["Age","Sex","Chest_pain","Resting_blood_pressure","Cholesterol",
                      "Fasting_blood_sugar","ECG_results","Maximum_heart_rate","Exercise_induced_angin",
                      "ST_depression","ST_slope","Major_vessels","Thalassemia_types"]
    
    dfH = pd.DataFrame(features_valueH, columns=features_nameH)
    outputH = modelH.predict(dfH)
        
    if outputH == 1:
        res_valH = "Heart Disease"
    else:
        res_valH = "no Heart Disease"
        

    return render_template('Heart_Form.html', prediction_text='Patient has {}'.format(res_valH))







# -------------Prediction for Breast Cancer----------------
@app.route('/predictB',methods=['POST'])
def predictBC():
    input_featuresB = [float(x) for x in request.form.values()]
    features_valueB = [np.array(input_featuresB)]
    
    features_nameB = ["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean",
                     "concavity_mean","concave_points_mean","symmetry_mean","radius_se","perimeter_se","area_se",
                     "compactness_se","concavity_se","concave points_se","fractal_dimension_se","radius_worst",
                     "texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst",
                     "concave points_worst","symmetry_worst","fractal_dimension_worst"]
    
    dfB = pd.DataFrame(features_valueB, columns=features_nameB)
    outputB = modelB.predict(dfB)
        
    if outputB == 1:
        res_valB = "Breast Cancer"
    else:
        res_valB = "no Breast Cancer"
        

    return render_template('Breast_Cancer_Form.html', prediction_text='Patient has {}'.format(res_valB))






# -------------Prediction for Diabetes----------------
@app.route('/predictD',methods=['POST'])
def predictDiabetes():
    input_featuresD = [float(x) for x in request.form.values()]
    features_valueD = [np.array(input_featuresD)]
    
    features_nameD = ["pregnancies","glucose","bloodpressure","skinthickness","insulin","bmi","dpf","age"]
    
    dfD = pd.DataFrame(features_valueD, columns=features_nameD)
    outputD = modelD.predict(dfD)
        
    if outputD == 1:
        res_valD = "Diabetic"
    else:
        res_valD = "Healthy"
        

    return render_template('Diabetes_form.html', prediction_text='Patient is {}'.format(res_valD))







# -------------Prediction for Kidney----------------
@app.route('/predictK',methods=['POST'])
def predictKidney():
    input_featuresK = [float(x) for x in request.form.values()]
    features_valueK = [np.array(input_featuresK)]
    
    features_nameK = ["age","bp","al","su","rbc","pc","pcc","ba","bgr","bu","sc","pot","wc","htn","dm","cad","pe","ane"]
    
    dfK = pd.DataFrame(features_valueK, columns=features_nameK)
    outputK = modelK.predict(dfK)
        
    if outputK == 1:
        res_valK = "kidney disease"
    else:
        res_valK = "no kidney disease"
        

    return render_template('Kidney_form.html', prediction_text='Patient has {}'.format(res_valK))









# -------------Prediction for liver----------------
@app.route('/predictL',methods=['POST'])
def predictLiver():
    input_featuresL = [float(x) for x in request.form.values()]
    features_valueL = [np.array(input_featuresL)]
    
    features_nameL = ["Age","Gender","Total_Bilirubin","Direct_Bilirubin","Alkaline_Phosphotase","Alamine_Aminotransferase",
                      "Aspartate_Aminotransferase","Total_Protiens","Albumin","Albumin_and_Globulin_Ratio"]
    
    dfL = pd.DataFrame(features_valueL, columns=features_nameL)
    outputL = modelL.predict(dfL)
        
    if outputL == 1:
        res_valL = "l̥īver disease"
    else:
        res_valL = "no liver disease "
        

    return render_template('Liver_Form.html', prediction_text='Patient has {}'.format(res_valL))






if __name__ == '__main__':
    app.run(debug = True)
