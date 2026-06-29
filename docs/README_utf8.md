# Run scheduling.py for listing then run link_to_details for alonhadat_details.csv then run main.py and you will have raw data with additional feature you should eda and cleaning for training model 
trước hết là chạy scaper\Alonhadat\scheduling.py để cào link từ 2 nguồn chính là (Hà nội và HCM)

Sau khi xong thì chạy scaper\Alonhadat\link_each_status.py 
để cáo các chi tiết theo link

Hiện tại còn phải chạy code trên kaggle (máy yếu quá)
https://www.kaggle.com/code/ltrngnhn/publicutilitypreprocessing

Finished merge density

next extract lat and lon for each row (in main)


done distance_to_center

done feature adding 

now validation data

task after scheduling for scraping

for poi_features get_nearest_poi (in 5000m radius) and count(in can be modify raidus)


