import json

#importing mbs.json file
with open("mbs.json","r") as read_files:
    data7=json.load(read_files)
#print(data7[0])
def mbs_data():
    mbs_model=[]
    for i in range(2):
         faculty_model.append(data7[i]['Year'])
    #list of availabl faculties
    print(mbs_model)

    
#for full info
def mbs_info(usermodel):
    usermodel = usermodel.upper()
    mbs_model = []
    for i in range(2):
        mbs_model.append(data7[i]['Year'])
    print(mbs_model)

    if usermodel in mbs_model:
        #index of usermodel
        x=mbs_model.index(usermodel)
        extract_model=data7[x]['Year']
        if extract_model==usermodel:
            print("KBC Bot: BBS "+ data7[x]['Year']+" year details:\n")
            print("Shift: "+data7[x]['Shift']+"\t\tYear Fee: "+data7[x]['Year_fee']+"\n")
            print("Course list:\n")
            for i in range(len(data7[x]['Subjects'])):
                print("*  "+data7[x]['Subjects'][i])
            if len(data7[x]['Electives'])>0:
                print("\nElective Subjects: (Choose at least one from below):\n")
                for i in range(len(data7[x]['Electives'])):
                    print("*  "+data7[x]['Electives'][i])

#for fee only
def year_fee(usermodel):
    usermodel = usermodel.upper()
    mbs_model = []
    for i in range(2):
        mbs_model.append(data7[i]['Year'])
    if usermodel in mbs_model:
        #index of usermodel
        x = mbs_model.index(usermodel)
        extract_model = data7[x]['Year']
        if extract_model == usermodel:
            print("\nKBC Bot: The fee for "+ data7[x]['Year'] +" year is "+ data7[x]['Year_fee'])


#z=input()
#mbs_info(z)
#year_fee(z)


