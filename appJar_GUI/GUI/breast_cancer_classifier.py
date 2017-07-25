import sys
sys.path.append("../../")
# import the library
from appJar import gui

# create a GUI variable called app
app = gui()


# the title of the button will be received as a parameter
def press(btn):
    app.setMessage("Classification result", """The image shows BENIGN breast cancer.""")


# add & configure widgets - widgets get a name, to help referencing them later
app.addLabel("title", "Breast Cancer Classifier")
app.setLabelBg("title", "light grey")





app.addLabelOptionBox("Choose cancer image", ["- Please select -", "img01", "img02",
                        "img03", "img04", "img05", "img06",
                        "img07", "img08"])

app.startLabelFrame("Image Preview")
#app.startLabelFrame("Image Preview", 0, 0)
app.addImage("simple", "..\images\SOB_B_A-14-22549AB-40-001.png")
app.stopLabelFrame()

app.addButton("Classify", press)

app.addEmptyMessage("Classification result")

app.setStretch("both")

app.setFont(14)

# start the GUI
app.go()



