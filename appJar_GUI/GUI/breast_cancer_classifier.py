import sys
sys.path.append("../../")
# import the library
from appJar import gui

# create a GUI variable called app
app = gui()


# the title of the button will be received as a parameter
def press(btn):
    app.setMessage("Classification result", """The image shows BENIGN breast cancer.""")

def browse(btn):
    app.openBox(title=None, dirName=None, fileTypes=None, asFile=False)

# add & configure widgets - widgets get a name, to help referencing them later
app.addLabel("title", "Breast Cancer Classifier")
app.setLabelBg("title", "light grey")



app.addButton("Choose Cancer Image", browse)

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



