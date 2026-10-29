def fill_missing_values(shipments,medians=None):
    if medians is None:
      medians = {"distance_km":
                 round(float(shipments['distance_km'].median()),2),
               "parcel_weight_kg":
                 round(float(shipments['parcel_weight_kg'].median()),2),
               "declared_value_inr":round(float(shipments['declared_value_inr'].median()),2)
               }
      #print(medians)
      for column,median_value in medians.items():
        shipments[column]=(shipments[column].fillna(median_value))
     
    return medians

def build_features(clean, hubs):
 result = clean.merge(hubs[
                              [
                               "hub_id",
                               "hub_city",
                               "region",
                               "hub_tier",
                               "hub_daily_capacity"
                               ]
                      ],
                      on="hub_id",
                      how="left",
                      validate="many_to_one")

 result["promised_hours"] = (result["service_type"]
                            .map(PROMISED_HOURS)
                            .astype(int)
                            )
 result["is_cod"]= (result["payment_mode"] == "COD").astype(int)
 result["is_weekend"]=(result["dispatch_time"].dt.day_of_week >=5 
                       ).astype(int)
 result["dispatch_hour"]=(result["dispatch_time"].dt.hour
                          ).astype(int)
 result["log_declared_value"]=np.log(result["declared_value_inr"])                                               
 #print(result)
 return result

def clean_shipments(raw):#, medians=None):
  print("Length of Raw ship data before deleting duplicate data:",len(raw))
  print("Duplicates rows in the Raw ship data:",raw.duplicated().sum())
  #drop exact duplicate rows as the duplicate data comprises 1% of the whole data
  shipments = raw.drop_duplicates().copy()
  print("Length of Raw ship data after deleting duplicate data:",len(shipments))
  print("Shape after duplicates:",shipments.shape)
  #strip leading and trailing spaces and convert to title case
  shipments["service_type"] = shipments["service_type"].str.strip().str.title()
  print("Unique service_type after string correction:",shipments["service_type"].unique())
  #To get count of shipments for each service type
  print("Number of shipments for each service type:",shipments["service_type"].value_counts())
  #Percentage of Contribution of each service type
  print("Percentage of Contribution of each service type:",shipments["service_type"].value_counts(normalize = True)*100)
  #To get count of missing values in column payment_mode
  print("To get count for each payment mode:",shipments["payment_mode"].value_counts())
  print("Count of missing values in column payment_mode: ",shipments.isnull().sum()['payment_mode'])
  #To get count of missing values in each column
  #print("Count of missing values in EACH column:\n",shipments.isnull().sum())
  #the total count of all missing values across the entire dataset
  print("The total count of all missing values across the entire dataset::",shipments.isnull().sum().sum())
  #payment_mode: replace a missing value with the placeholder "Unknown";
  shipments["payment_mode"]=shipments["payment_mode"].fillna("Unknown")
  #To get count of missing values in column payment_mode
  print("After update Count of missing values in column payment_mode: ",shipments.isnull().sum()['payment_mode'])
  #declared_value: remove the rupee sign and the thousands commas, convert to a number, store it in a new column declared_value_inr
  shipments["declared_value_inr"]=pd.to_numeric(shipments["declared_value"].str.replace('₹','').str.replace(',',''))
  #drop the text column
  shipments.drop('declared_value',axis=1,inplace=True)
  #dispatch_time: convert to a datetime
  shipments['dispatch_time'] = pd.to_datetime(shipments['dispatch_time'],errors='coerce')
  #To get count of missing values in column parcel_weight_kg
  print("Before update Count of missing values in column parcel_weight_kg: ",shipments.isnull().sum()['parcel_weight_kg'])
  count = len(shipments[shipments['parcel_weight_kg']>100])
  print("Count of parcel weight greater than 100::",count)
  shipments.loc[(shipments['parcel_weight_kg'] > 100),'parcel_weight_kg']=np.nan
    #To get count of missing values in each column
  print("Count of missing values in EACH column:\n",shipments.isnull().sum())
  medians = fill_missing_values(shipments)
     #To get count of missing values in each column
  print("After correction values in EACH column:\n",shipments.isnull().sum())
  
  
  return shipments


raw_ship_df_cleaned = clean_shipments(raw_ship_df)
featured_df=build_features(raw_ship_df_cleaned,hubs)
featured_df.head()
#hubs_df
#new_ship_df
#raw_ship_df_cleaned.info()
#raw_ship_df_cleaned['dispatch_time'].dtype#to validate the data type
