import json

#importing bbm.json file
with open("bbm.json", "r") as read_file:
    data5 = json.load(read_file)
#print(data5[0])

def bbm_data():
    bbm_model = []
    for i in range(8):
        faculty_model.append(data5[i]['Semester'])
    #list of available faculties
    print(bbm_model)


#for full info
def bbm_info(usermodel):
    usermodel = usermodel.upper()
    bbm_model = []
    for i in range(8):
        bbm_model.append(data5[i]['Semester'])
    #print(bbm_model)
                
    if usermodel in bbm_model:
        #index of usermodel
        x = bbm_model.index(usermodel)
        extract_model = data5[x]['Semester']
        if extract_model == usermodel:
            print("KBC Bot: BBM "+ data5[x]['Semester']+" semester details:\n")
            print("Shift: "+data5[x]['Shift']+"\t\tSemester Fee: "+data5[x]['Semester_fee']+"\n")
            print("Course list:\n")
            for i in range(len(data5[x]['Subjects'])):
                print("*  "+data5[x]['Subjects'][i])
            if len(data5[x]['Electives'])>0:
                print("\nElective Subjects: (Choose at least one from below):\n")
                for i in range(len(data5[x]['Electives'])):
                    print("*  "+data5[x]['Electives'][i])

#for fee only
def semester_fee(usermodel):
    usermodel = usermodel.upper()
    bbm_model = []
    for i in range(8):
        bbm_model.append(data5[i]['Semester'])
    if usermodel in bbm_model:
        #index of usermodel
        x = bbm_model.index(usermodel)
        extract_model = data5[x]['Semester']
        if extract_model == usermodel:
            print("KBC Bot: The fee for "+ data5[x]['Semester'] +" semester is "+ data5[x]['Semester_fee'])


#z=input()
#bbm_info(z)
#semester_fee(z)
