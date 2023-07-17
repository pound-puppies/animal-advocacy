# Animal-Advocacy-Project
### Brian O'Neil, Scott Barnett, Keila Camarillo, and Esayas Asefa
### 10 July 2023
## Project description with goals
### Description
* Using the Austin Animal Center data from 2013 to present, our team will create a model to best predict whether an cat or dog will be adopted. The purpose is to give insight to animal shelters that can use the model as a tool to decide where to focus resources to increase adoption rates. The key is early intervention for cats/dogs to increase adoption resources on those with lower rates of adoption. 

### Goals¶
* Discover drivers of outcome
* Use drivers of outcomes to develop machine learning models to predict outcomes

### Initial Thoughts
* Our initial hypothesis is that the drivers of outcome will be breed, age, condition, species, and sex.

## Initial hypotheses and/or questions you have of the data, ideas

- Is Month Related to Outcome?
- Is Breed Related to Outcome?
- Is Sex Related to Outcome?
- Is Species Related to Outcome?
- Is Condition Related to Outcome

*****************************************
## Project Plan 
* Data acquired and join were from Austin Animal Center
    * Files were downloaded and converted to dataframes from xls format
* The data was aquired on `10 July 2023`
* Two datasets downloaded from data.austintexas.gov: [Intake Dataset](https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Intakes/wter-evkm) & [Outcome Dataset](https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Outcomes/9t4d-g238)
* Intake Dataset: 153,077 rows and 12 columns 
* Outcome Dataset: 153,449 row and 12 columns
* Each row represents an animal's case
* Each column represents features of the animal


### Draw conclusions
* June, July, August, and December had higher volumes
* Mixed breeds are more likely to be transfered or adopted
* Fixed animals are far more likely to be adopted
* Cats are slightly more likely to have an ‘other’ or transfer outcome than dogs
* Cats and dogs with normal conditions are more likely to be adopted.

## Data Dictionary
|Feature   |Datatype| Unit       |Description   |
|----------|--------|------------|--------------|
|Source ID | String |Alphanumeric |Person who initiated the intake.|
|Animal ID | String |Alphanumeric | Unique number assigned to each animal when their record is created in the database |
|Animal Type| String| Alphanumeric|Animal category: dog, cat, wildlife, other, etc.  |
|Activity Number|Unique number assigned to an activity related to a service request.|
|Activity Sequence|Sequence starts with 1 usually then a follow up sequence is created until activity is completed.|
|Census Tract| | |Government census tract in which the action was located.|
|Council District|City of Dallas Council District in which the action was located.|
|DateTime| | | | |Date and time of action  |
|MonthYear|Month and year of aciton|
|Kennel Number|Location of the animal at the time of the report|
|Kennel Status|Availability of the animal.|
|Intake Type|Type or purpose of intake; used primarily to analyze intake trends.|
|Intak Total|Additional categorization of purpose of intake; used primarily to analyze intake trends.|
|Reason|Reason the animal was surrendered or taken in.|
|Staff Id|Unique ID number assigned to the staff person who entered the record.|
|Intake Date|Date the animal was intaken by DAS.|
|Intake Time|Time the animal was intaken by DAS.|
|Due Out|Date the animal's stray hold expires and animal will be available for non-return to owner outcomes; date DAS has full ownership of the animal based on city ordinance.|
|Intake Condition|Apparent medical condition of the animal when it was taken in by DAS.|
|Hold Request|Routing or pathway identified for the animal at the time of the report. Pathways are used to move animals towards the outcome management recommends at the time based on behavior, medical condition, and history. Pathways are reviewed and updated frequently as an animal's behavior or medical condition changes.|
|Outcome Type|Final outcome of the animal if they are no longer under the care of DAS at the time of the report.|
|Outcome Subtype|Additional details on the outcome of the animal used primiarly for outcome trend analysis.|
|Outcome Date| Date the animal was outcomed by DAS / left DAS' care.|
|Outcome Time|Time the animal was outcomed by DAS / left DAS' care.|
|Receipt number|Unique number assigned to each financial transaction that occurs in Chameleon database.|
|Impound Number|Unique number assigned to each impound performed by DAS staff; each impound can include multiple animals.|
|Service Request Number|Unique number assigned to each impound performed by DAS staff; each impound can include multiple animals.|
|Outcome condition|Apparent medical condition of the animal when it was released from DAS.|
|Chip status|Notates whether staff were successful in scanning animal for a microchip.|
|Animal origin| Notates whether the animal came in through DAS' Pet Support Lobby (Over the Counter) or through Field Services (Field).|
|Additional Information|Additional staff notes.|
|Month|Month the record was created.|
|Year |City of Dallas Fiscal Year the record was created.|
|Date of Birth | Birth date of the animal  |
|Sex upon outcome | Whether the animal was neutered/spayed during outcome  |
|Age upon outcome | Age of animal at time of outcome|
|Breed |Breed of animal|
|Color |The color of the animal|

## Steps to Reproduce
* 1. Clone this repo: git@github.com:pound-puppies/animal-advocacy.git
* 2. Go to team [Google Drive link here:](https://drive.google.com/drive/folders/1hV0WQezLiQpS06MIc0Kggy8Iq0mdoTLh) 
* 3. Download austin_animal_intakes.csv and austin_animal_outcomes.csv and put in cloned personal repository
* 4. Run notebook.

## Takeaways and Conclusions
- Identifed features that have a significant relationship with outcome:
    * June, July, August, and December had higher volumes
    * Mixed breeds are more likely to be transfered or adopted
    * Fixed animals are far more likely to be adopted
    * Cats are slightly more likely to have an ‘other’ or transfer outcome than dogs
    * Cats and dogs with normal conditions are more likely to be adopted.
    
- Month of outcome, Breed of species, Sex, Species, Condition, Mix_color, Month_Rel:
    * Each feature had a statistically significant relaitonship with outcome

** Year_rel: Showed overall trend and would not be a accurate prediction**
** Outcome_age: Data integrity issued was raised when we found negative ages**

# Recommendations

* We have data governance recommendations:
    - We found tens of thousands of rows with data that was missing and/or had nonsensical information
    * Collect information more information such as: 
        - Incidents (e.g. biting, abuse hx) 
        - Stated reason for return 
        - Reasons for turn in by owners
        - Vaccination status
        - Who turned the animal in (e.g. citizen, law enforcement, organization)

# Next Steps
* If provided more time we would use NLP to review the polarity of the names. 
* Review other shelters with different features

s