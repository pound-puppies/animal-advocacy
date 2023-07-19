# Animal-Advocacy-Project
### Brian O'Neil, Scott Barnett, Keila Camarillo, and Esayas Asefa
### 10 July 2023

<center><h1>Animal Advocacy</center>

<a name ='toc'></a>
# Table of Contents 
0. [Domain Context](#domain_context)
    1. [](#Animal_disc)
1. [Project Description](#project_Description)
    1. [Project Objectives](#project_objectives)
    3. [Deliverables](#deliverables)
2. [Executive Summary](#exe_sum)
    1. [Goals](#goals)
    2. [Findings](#findings)
3. [Acquire Data](#acquire)
    1. [Data Dictonary](#data_dict)
    2. [Acquire Takeaways](#acquire_takeaways)
4. [Prepare Data](#prep_data)
    1. [Distributions](#distributions)
    2. [Prepare Takeaways](#prepare_takeaways)
5. [Data Exploration](#explore)
    1. [Explore Takeaways](#explore_takeaways)
    2. [Hypothesis](#hypothesis)
6. [Modeling & Evaluation](#modeling)
    1. [Modeling Takeaways](#model_takeaways)
7. [Project Delivery](#delivery)
    1. [Conclusions & Next Steps](#conclusions_next_steps)
    2. [Project Replication](#replication)
   
<hr style="border-top: 10px groove tan; margin-top: 5px; margin-bottom: 5px"></hr>

    
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

## Prepare
* Two dataframes were created, model_df with encoded variables
* Altered column names for readability, and convenience
    - We removed spaces and capital letters
* Dropped nulls, duplicate ids, species other than cats and dogs
    - 12000 unique animal_id's were repeat offenders/duplicates (had been in the shelter more than once)
        - Their "intake date" however was input based on the first time they came into the shelter, in every instance of their intake
            - After discussion we decided to remove all animal_id's that had duplicates
    - We removed non-cats and non-dogs
        - We decided to it was impractical to keep animals that were uncommon to have as pets or see in urban settings
* Converted data types of various columns to appropriate ones such as 'dates' from string to datetime
* Columns and feature categories were renamed:
    Renamed Columns:
        - "outcome_type" was renamed to "outcome" (target variable)
        - Our original dataset had intake names, and outcome names for each animal when they'd rarely changed
            - We decided to keep their intake name column
        - "Animal type_x" to "species"
        - monthyear_x, and monthyear_y removed and 2 columns, one for month and one for year were created
       Feature Changes:
        - Conditions:
            - We changed 10 intake conditions to 6, based on what type of care they may need and survivability
        - Outcome month and year:
            - We decided after engineering features from our date based columns to only keep outcome month, year (both noted as rel_month, and rel_year), and age 
        - Breed:
            - Changed to "mix", "two_breeds", and "pure_breed"
                - There were 2200 
        - Outcome was changed from 
* Engineered Features: outcome, breed, outcome_age, primary_color, is_tabby, mix_color
    - Outcome was changed from 10 different outcomes to 3
        - If animals were placed in a home, they were categorized as "Adopt"
        - If animals were transfered to another facility they were categorized as "Transfer"
        - If animals had any other outcome they were categorized as "Other"
    - Outcome age was more important to us than the animals intake age, since we care primarily about what effects OUTCOMES for the animal
    - Primary color, is_tabby, and mix_color columns allowed to more easily boolean the feature than to have 4000 different colors
* Removed Features below: 
    - id - After removing all duplicates, no longer added value
    - name_x - names never/rarely changed
    - monthyear_x - after calculating age of outcome tenure_days (length of stay), decided it was unnecessary
    - sex upon intake - sex upon outcome was more appropriate
    - found location - we deemed this an unnecessary feature for our model
    - outcome subtype - only several hundred rows had outcome subtypes, we removed it 
    - intake_datetime - after calculating age at outcome, tenure_days (length of stay), and found it to be unreliable in thousands of rows since we were getting negative tenure days
* Split data into train, validate and test (approx. 60/20/20), stratifying on 'outcome'
* Outliers were not adressed as they were part of the target


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