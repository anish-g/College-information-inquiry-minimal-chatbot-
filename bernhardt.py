import json

#importing bernhardt.json file
with open("bernhardt.json", "r") as read_file:
    data1 = json.load(read_file)
#print(data1[0]['Name'])

def bernhardt_data():
    print("KBC Bot: "+data1[0]['Name']+"\n"+data1[0]['Address']+"\tPhone: "+data1[0]['Contact'])
    print("Website: "+data1[0]['Website']+"\tEmail: "+data1[0]['Email']+"\n\n"+data1[0]['Intro'])
    print("\n--------------------------------------------------------------- Vision ---------------------------------------------------------------\n"+data1[0]['Vision'])
    print("\n--------------------------------------------------------------- Mission --------------------------------------------------------------- \n"+data1[0]['Mission'])
    
    


   



    
#bernhardt_data()
