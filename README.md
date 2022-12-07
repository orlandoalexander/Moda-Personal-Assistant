Models folder with python file for each model

Each file should contain a class with the name of the type (and the name of the file)

Add General notes to the top of this file and notes about your model in this file (below):

## Attributes

**1. Design**
  - 7 Categories:
    - Floral
    - Graphic
    - Striped
    - Embroidered
    - Pleated
    - Solid
    - Lattice (Plaid)


**2. Sleeves**
  - Can take the full dataframe (data_full) but would it be best to pass it a
  prefiltered dataframe? (Without other columns and 'lower)
  - Same question for the other Models
  - 3 Categories:
    - Long
    - Short
    - Sleeveless


**3. Length**

  - Should be passed only dresses for speed and to drop 'no dress' column. Otherwise performing unnecessary train/predict tasks with another feature, more data.
  - 2 Categories:
    - Maxi
    - Mini

**4. Part (Let's rename this "Neckline")**

  - Only distinguishes necklines, so should be renamed and again shouldn't be passed 'lower'
  - 4 Categories:
    - Crew
    - V
    - Square
    - None

**5. Fabric**
    - 6 Categories:
      - Denim
      - Chiffon
      - Cotton
      - Leather
      - Faux
      - Knit

**6. Fit**
    - 3 Categories:
      - Tight
      - Loose
      - Conventional
    
## Categories
**46 categories**
 
