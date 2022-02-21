# from collections import Counter
# my_str = "I own a Cayenne S 2019. My battery went dead, after 10 days at the dealership, my car still has not been fixed. Porsche doesn't have a battery for this car available in the USA. Cannot believe this. It is my third Porsche in three years. It will be my last one. How is possible that a company like Porsche do not has batteries in this inventory in the USA. Very dissatisfaction for the customer service as well. Really bad."
# myList = my_str.split()
# print (dict(Counter(myList)))
import pandas as pd

my_dict = {'cayenne': 1, 'battery': 3, 'go': 1, 'dead': 1, 'day': 1, 'dealership': 1, 'car': 2, 'fix': 1, 'porsche': 3, 'not': 1, 'available': 1, 'usa': 2, 'believe': 1, 'year': 1, 'possible': 1, 'company': 1, 'like': 1, 'inventory': 1, 'dissatisfaction': 1, 'customer': 1, 'service': 1, 'bad': 1}

df = pd.DataFrame({
                    'Word':list(my_dict.keys()), 
                    'Frequency':list(my_dict.values())
                })
df = df.sort_values(by='Frequency', ascending=False).head(10)
print(df)

