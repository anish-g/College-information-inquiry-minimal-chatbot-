import json

#importing csit.json file
with open("csit.json", "r") as read_file:
    data3 = json.load(read_file)
#print(data3[0])

def csit_data():
    csit_model = []
    for i in range(8):
        faculty_model.append(data3[i]['Semester'])
    #list of available faculties
    print(csit_model)


#for full info
def csit_info(usermodel):
    usermodel = usermodel.upper()
    csit_model = []
    for i in range(8):
        csit_model.append(data3[i]['Semester'])
    #print(csit_model)
                
    if usermodel in csit_model:
        #index of usermodel
        x = csit_model.index(usermodel)
        extract_model = data3[x]['Semester']
        if extract_model == usermodel:
            print("KBC Bot: BSc.CSIT "+ data3[x]['Semester']+" semester details:\n")
            print("Shift: "+data3[x]['Shift']+"\t\tSemester Fee: "+data3[x]['Semester_fee']+"\n")
            print("Course list:\n")
            for i in range(len(data3[x]['Subjects'])):
                print("*  "+data3[x]['Subjects'][i])
            if len(data3[x]['Electives'])>0:
                print("\nElective Subjects: (Choose at least one from below):\n")
                for i in range(len(data3[x]['Electives'])):
                    print("*  "+data3[x]['Electives'][i])

#for fee only
def semester_fee(usermodel):
    usermodel = usermodel.upper()
    csit_model = []
    for i in range(8):
        csit_model.append(data3[i]['Semester'])
    if usermodel in csit_model:
        #index of usermodel
        x = csit_model.index(usermodel)
        extract_model = data3[x]['Semester']
        if extract_model == usermodel:
            print("KBC Bot: The fee for "+ data3[x]['Semester'] +" semester is "+ data3[x]['Semester_fee'])


#z=input()
#csit_info(z)
#semester_fee(z)
