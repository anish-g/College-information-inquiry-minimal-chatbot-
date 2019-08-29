import json
#importing bca.json file
with open("bca.json","r") as read_file:
    data4=json.load(read_file)
#print(data4[0])


def bca_data():
    bca_model=[]
    for i in range (7):
        bca_model.append(data4[i]['Semester'])
    #list of bca semesters
    print(bca_model)

#for full info about BCA

def bca_info(usermodel):
    usermodel = usermodel.upper()
    bca_model = []
    for i in range(8):
        bca_model.append(data4[i]['Semester'])
    #print(bca_model)

    if usermodel in bca_model:
         #index of usermodel
        x = bca_model.index(usermodel)
        extract_model = data4[x]['Semester']
        if extract_model == usermodel:
            print("KBC Bot: BCA "+ data4[x]['Semester']+" semester details:\n")
            print("Shift: "+data4[x]['Shift']+"\t\tSemester Fee: "+data4[x]['Semester_fee']+"\n")
            print("Course list:\n")
            for i in range(len(data4[x]['Subjects'])):
                print("*  "+data4[x]['Subjects'][i])
            if len(data4[x]['Electives'])> 0:
                print("\nElective Subjects: (Choose at least one from below):\n")
                for i in range(len(data4[x]['Electives'])):
                    print("*  "+data4[x]['Electives'][i])

#for fee only
def semester_fee(usermodel):
    usermodel = usermodel.upper()
    bca_model = []
    for i in range(8):
        bca_model.append(data4[i]['Semester'])
    if usermodel in bca_model:
        #index of usermodel
        x = bca_model.index(usermodel)
        extract_model = data4[x]['Semester']
        if extract_model == usermodel:
            print("KBC Bot: The fee for "+ data4[x]['Semester'] +" semester is "+ data4[x]['Semester_fee'])


#z=input()
#bca_info(z)
#semester_fee(z)



        



     
