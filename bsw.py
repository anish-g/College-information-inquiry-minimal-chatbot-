import json

#importing bsw.json file
with open("bsw.json","r") as read_files:
    data8=json.load(read_files)
#print(data8[0])
def bsw_data():
    bsw_model=[]
    for i in range(3):
         faculty_model.append(data8[i]['Year'])
    #list of availabl faculties
    print(bsw_model)

    
#for full info
def bsw_info(usermodel):
    usermodel = usermodel.upper()
    bsw_model = []
    for i in range(3):
        bsw_model.append(data8[i]['Year'])
    #print(bsw_model)

    if usermodel in bsw_model:
        #index of usermodel
        x=bsw_model.index(usermodel)
        extract_model=data8[x]['Year']
        if extract_model==usermodel:
            print("KBC Bot: BSW "+ data8[x]['Year']+" year details:\n")
            print("Shift: "+data8[x]['Shift']+"\t\tYear Fee: "+data8[x]['Year_fee']+"\n")
            print("Course list:\n")
            for i in range(len(data8[x]['Subjects'])):
                print("*  "+data8[x]['Subjects'][i])
            if len(data8[x]['Electives'])>0:
                print("\nElective Subjects: (Choose at least one from below):\n")
                for i in range(len(data8[x]['Electives'])):
                    print("*  "+data8[x]['Electives'][i])

#for fee only
def year_fee(usermodel):
    usermodel = usermodel.upper()
    bsw_model = []
    for i in range(3):
        bsw_model.append(data8[i]['Year'])
    if usermodel in bsw_model:
        #index of usermodel
        x = bsw_model.index(usermodel)
        extract_model = data8[x]['Year']
        if extract_model == usermodel:
            print("\nKBC Bot: The fee for "+ data8[x]['Year'] +" year is "+ data8[x]['Year_fee'])


#z=input()
#bsw_info(z)
#year_fee(z)


