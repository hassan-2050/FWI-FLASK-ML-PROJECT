from flask import Flask,request,render_template,jsonify
import pickle
application = Flask(__name__)
app=application
scaler=pickle.load(open("models/scaler.pkl","rb"))
modeler=pickle.load(open("models/ridge.pkl","rb"))
@app.route("/")
def hello_world():

    return render_template("index.html")



@app.route("/predict",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="POST":
        temp=float(request.form.get("Temperature"))
        rh=float(request.form.get("RH"))
        ws=float(request.form.get("Ws"))
        rain=float(request.form.get("Rain"))
        ffmc=float(request.form.get("FFMC"))
        dmc=float(request.form.get("DMC"))
        isi=float(request.form.get("ISI"))
        classes=float(request.form.get("Classes"))
        region=float(request.form.get("Region"))
        lists=[temp,rh,ws,rain,ffmc,dmc,isi,classes,region]
        print(lists)
        scaled=scaler.transform([lists])
        model=modeler.predict(scaled)
        print(model)
        return render_template("home.html",result=model[0])
    else:    
        return render_template("home.html")



if __name__=="__main__":
    app.run(host="0.0.0.0")
