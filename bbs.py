import json

#importing bbs.json file
with open("bbs.json","r") as read_files:
    data6=json.load(read_files)
#print(data6[0])
def bbs_data():
    bbs_model=[]
    for i in range(4):
         faculty_model.append(data6[i]['Year'])
    #list of availabl faculties
    print(bbs_model)

    
#for full info
def bbs_info(usermodel):
    usermodel = usermodel.upper()
    bbs_model = []
    for i in range(4):
        bbs_model.append(data6[i]['Year'])
    print(bbs_model)

    if usermodel in bbs_model:
        #index of usermodel
        x=bbs_model.index(usermodel)
        extract_model=data6[x]['Year']
        if extract_model==usermodel:
            print("KBC Bot: BBS "+ data6[x]['Year']+" year details:\n")
            print("Shift: "+data6[x]['Shift']+"\t\tYear Fee: "+data6[x]['Year_fee']+"\n")
            print("Course list:\n")
            for i in range(len(data6[x]['Subjects'])):
                print("*  "+data6[x]['Subjects'][i])
            if len(data6[x]['Electives'])>0:
                print("\nElective Subjects: (Choose at least one from below):\n")
                for i in range(len(data6[x]['Electives'])):
                    print("*  "+data6[x]['Electives'][i])

#for fee only
def year_fee(usermodel):
    usermodel = usermodel.upper()
    bbs_model = []
    for i in range(4):
        bbs_model.append(data6[i]['Year'])
    if usermodel in bbs_model:
        #index of usermodel
        x = bbs_model.index(usermodel)
        extract_model = data6[x]['Year']
        if extract_model == usermodel:
            print("\nKBC Bot: The fee for "+ data6[x]['Year'] +" year is "+ data6[x]['Year_fee'])


#z=input()
#bbs_info(z)
#year_fee(z)


