#!/usr/bin/env python
# coding: utf-8

# # Data mapping

# In this session, we will build on all we have learnt so far about loading and manipulating (spatial) data and apply it to one of the most commonly used forms of spatial analysis: choropleths. Remember these are maps that display the spatial distribution of a variable encoded in a color scheme, also called *palette*. Although there are many ways in which you can convert the values of a variable into a specific color, we will focus in this context only on a handful of them, in particular:
# 
# * Unique values.
# * Equal interval.
# * Quantiles.
# * Fisher-Jenks.
# 
# In addition, we will cover how to add base maps that provide context from rasters and, in two optional extensions, will review two more additional ways of displaying data in maps: cartograms and conditional maps.
# 
# Before all this mapping fun, let us get the importing of libraries and data loading out of the way:

# In[3]:


# get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
import pandas as pd
from pysal.viz import mapclassify
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt


# ## Data
# 
# For this tutorial, we will use the recently released 2015 Index of Multiple Deprivation (IMD) for England and Wales. This dataset can be most easily downloaded from the CDRC data store ([link](https://data.cdrc.ac.uk/dataset/cdrc-english-indices-of-deprivation-2015-geodata-pack-liverpool-e08000012)) and, since it already comes both in tabular as well as spatial data format (shapefile), it does not need merging or joining to additional geometries.
# 
# Although all the elements of the IMD, including the ranks and the scores themselves, are in the IMD dataset, we will also be combining them with additional data from the Census, to explore how deprivation is related to other socio-demographic characteristics of the area. For that we will revisit the Census Data Pack ([link](https://data.cdrc.ac.uk/dataset/cdrc-2011-census-data-packs-for-local-authority-district-liverpool-e08000012)) we used previously.
# 
# In order to create maps with a base layer that provides context, we will be using a raster file derived from [OS VectorMap District (Backdrop Raster)](https://www.ordnancesurvey.co.uk/business-and-government/products/vectormap-district.html) and available for download on [this link](http://darribas.org/gds19/content/labs/figs/lab04_liverpool.tif).
# 
# As usual, let us set the paths to the folders containing the files before anything so we can then focus on data analysis exclusively (keep in mind the specific paths will probably be different for your computer):

# In[7]:


# This will be different on your computer and will depend on where
# you have downloaded the files
imd_shp = '../data/E08000012/shapefiles/E08000012.shp'
liv_path = '../data/lab04_liverpool.tif'
data_path = '../data/Liverpool/'


# **IMPORTANT**: the paths above might have look different in your computer. See [this introductory notebook](begin.html) for more details about how to set your paths.

# * **IMD data**
# 
# Now we can load up the IMD data exactly as we did earlier for a shapefile:

# In[9]:


# Read the file in
imd = gpd.read_file(imd_shp)
# Index it on the LSOA ID
imd = imd.set_index('LSOA11CD')
# Display summary
imd.info()


# Note how on line 4 we *index* the resulting table `imd` with the column `LSOA11CD`. Effectively, this means we are "naming" the rows, the same way we the columns are named, using the column `LSOA11CD`, which contains the unique ID's of each area. This affords us some nice slicing and querying capabilities as well as permitting to merge the table with other ones more easily. 
# 
# Pay attention also to how exactly we index the table: we create a new object that is named in the same way, `imd`, but that contains the result of applying the function `set_index` to the original object `imd`. As usual, there are many ways to index a table in Python, but this is one of the most direct and expressive ones.

# * **Census data**
# 
# In order to explore additional dimensions of deprivation, and to have categorical data to display with "unique values" choropleths, we will use some of the Census data pack. Although most of the Census variables are continuous, we will transform them to create *categorical* characteristics. Remember a categorical variable is one that comprises only a limited number of potential values, and these are not comparable with each other across a numerical scale. For example, religion or country of origin are categorical variables. It is not possible to compare their different values in a quantitative way (religion A is not double or half of religion B) but instead they represent qualitative differences.
# 
# In particular, we are going to use tables `QS104EW` (Gender) and `KS103EW` (marital status). The way these are presented in its raw form is as tabulated counts of each of the possible categories. Our strategy to turn these into a single categorical variable for each case is to compare the counts for each area and assign that of the largest case. For example, in the first case, an area will be labelled as "male" if there are more males than females living in that particular LSOA. In the case of marital status, although there are more cases, we will simplify and use only "married" and "single" and assign one or the other on the bases of which ones are more common in each particular area.
# 
# **NOTE**: the following code snippet involves some data transformations that are a bit more advanced that what is covered in this course. Simply run them to load the data, but you are not expected to know some of the coding tricks required in this cell.

# In[10]:


# Gender breakup
# Read table (csv file)
gender = pd.read_csv(data_path+'tables/QS104EW_lsoa11.csv', index_col='GeographyCode')
# Rename columns from code to human-readable name
gender = gender.rename(columns={'QS104EW0002': 'Male',                                 'QS104EW0003': 'Female'})[['Male', 'Female']]
# Create male-female switcher
maj_male = gender['Male'] > gender['Female']
# Add "Gender_Majority" variable to table and assign the switcher
gender['Gender_Majority'] = maj_male
# Replace `True` values with "Male" and `False` with "Female"
gender.loc[gender['Gender_Majority']==True, 'Gender_Majority'] = 'Male'
gender.loc[gender['Gender_Majority']==False, 'Gender_Majority'] = 'Female'

# Status breakup
# Read table (csv file)
sinmar = pd.read_csv(data_path+'tables/KS103EW_lsoa11.csv', index_col='GeographyCode')
# Rename columns from code to human-readable name
sinmar = sinmar.rename(columns={'KS103EW0002': 'Single',                                 'KS103EW0003': 'Married'})[['Single', 'Married']]
# Create sigle-married switcher
maj_sin = sinmar['Single'] > sinmar['Married']
# Add "Status_Majority" variable to table and assign the switcher
sinmar['Status_Majority'] = maj_sin
# Replace `True` values with "Single" and `False` with "Married"
sinmar.loc[sinmar['Status_Majority']==True, 'Status_Majority'] = 'Single'
sinmar.loc[sinmar['Status_Majority']==False, 'Status_Majority'] = 'Married'

# Join
both = imd.join(sinmar).join(gender)
# Reset the CRS after join
both.crs = imd.crs


# This creates the table we will be using for the rest of the session:

# In[11]:


both.head()


# A look at the variables reveals that, in effect, we have successfuly merged the IMD data with the categorical variables derived from Census tables:

# In[12]:


both.info()


# Now we are fully ready to map!

# ## Choropleths

# ### Unique values

# A choropleth for categorical variables simply assigns a different color to every potential value in the series. The main requirement in this case is then for the color scheme to reflect the fact that different values are not ordered or follow a particular scale.
# 
# In Python, thanks to `geopandas`, creating categorical choropleths is possible with one line of code. To demonstrate this, we can plot the spatial distribution of LSOAs with a more female population than male and viceversa:

# In[16]:


both.plot(column='Gender_Majority', 
          categorical=True, 
          legend=True, linewidth=0.1)


# Let us stop for a second in a few crucial aspects:
# 
# * Note how we are using the same approach as for basic maps, the command `plot`, but we now need to add the argument `column` to specify which column in particular is to be represented.
# * Since the variable is categorical we need to make that explicit by setting the argument `categorical` to `True`.
# * As an optional argument, we can set `legend` to `True` and the resulting figure will include a legend with the names of all the values in the map.
# * Unless we specify a different colormap, the selected one respects the categorical nature of the data by not implying a gradient or scale but a qualitative structure.

# ---
# 
# **[Optional exercise]**
# 
# Create a categorical map of the marital status in Liverpool. Where are the areas with more married than single population?
# 
# ---

# In[17]:


both.plot(column='Status_Majority', 
          categorical=True, 
          legend=True, linewidth=0.1)


# ### Equal interval

# If, instead of categorical variables, we want to display the geographical distribution of a continuous phenomenon, we need to select a way to encode each value into a color. One potential solution is applying what is usually called "equal intervals". The intuition of this method is to split the *range* of the distribution, the difference between the minimum and maximum value, into equally large segments and to assign a different color to each of them according to a palette that reflects the fact that values are ordered.
# 
# Using the example of the position of a LSOA in the national ranking of the IMD (`imd_rank`), we can calculate these segments, also called bins or buckets, using the library `PySAL`:

# In[20]:


classi = mapclassify.EqualInterval(imd['imd_rank'], k=7)
classi


# The only additional argument to pass to `Equal_Interval`, other than the actual variable we would like to classify is the number of segments we want to create, `k`, which we are arbitrarily setting to seven in this case. This will be the number of colors that will be plotted on the map so, although having several can give more detail, at some point the marginal value of an additional one is fairly limited, given the ability of the brain to tell any differences.
# 
# Once we have classified the variable, we can check the actual break points where values stop being in one class and become part of the next one:

# In[21]:


classi.bins


# The array of breaking points above implies that any value in the variable below 4604.9 will get the first color in the gradient when mapped, values between 4604.9 and 9185.7 the next one, and so on.
# 
# The key characteristic in equal interval maps is that the bins are allocated based on the magnitude on the values, irrespective of how many obervations fall into each bin as a result of it. In highly skewed distributions, this can result in bins with a large number of observations, while others only have a handful of outliers. This can be seen in the submmary table printed out above, where 156 LSOAs are in the first group, but only five of them belong to the one with highest values. This can also be represented visually with a kernel density plot where the break points are included as well:

# In[22]:


# Set up the figure
f, ax = plt.subplots(1)
# Plot the kernel density estimation (KDE)
sns.kdeplot(imd['imd_rank'], shade=True)
# Add a blue tick for every value at the bottom of the plot (rugs)
sns.rugplot(imd['imd_rank'], alpha=0.5)
# Loop over each break point and plot a vertical red line
for cut in classi.bins:
    plt.axvline(cut, color='red', linewidth=0.75)
# Display image
plt.show()


# Technically speaking, the figure is created by overlaying a KDE plot with vertical bars for each of the break points. This makes much more explicit the issue highlighed by which the first bin contains a large amount of observations while the one with top values only encompasses a handful of them.
# 
# To create a map that displays the colors assigned by the equal interval classification algorithm, we use a similar approach as with unique values but with some key differences:

# In[35]:


imd.plot(column='imd_rank', scheme='equal_interval', k=7, 
         cmap=plt.cm.Blues_r, alpha=1,
         edgecolor='w', linewidth=0.1)


# Pay attention to the key differences:
# 
# * Instead of specifyig `categorical` as `True`, we replace it by the argument `scheme`, which we will use for all choropleths that require a continuous classification scheme. In this case, we set it to `equal_interval`.
# * As above, we set the number of colors to 7. Note that we need not pass the bins we calculated above, the plotting method does it itself under the hood for us.
# * As optional arguments, we can change the colormap to a blue gradient, which is one of the recommended ones by [ColorBrewer](http://colorbrewer2.org/) for a sequential palette. **NOTE** also how we use an appropriate palette: `imd_rank` goes from most to least deprived to, so we apply a palette (`Blues_r`, where the `_r` stands for reverse) for which the smaller values are encoded in darker blue.
# * Equally optional, some of the arguments we learned with basic maps, such as the degree of transparency, also apply in this context.
# 
# Substantively, the map also makes very explicit the fact that many areas are put into the same bin as the amount of white polygons is very large.

# ---
# 
# **[Optional exercise]**
# 
# Create an equal interval kde plot and map of the actual score of the IMD (`imd_score`). Is the same palette appropriate?
# 
# As a bonus, try including a legend in the map, following a similar approach as in unique values maps.
# 
# ---

# In[37]:


f, ax = plt.subplots(1, figsize=(15,10))
imd.plot(ax=ax, column='imd_score', scheme='equal_interval', k=7, 
         cmap=plt.cm.Greens, alpha=1,
         legend=True,
         edgecolor='w', linewidth=0.1)
plt.axis('equal')
plt.show()


# ### Quantiles

# One solution to obtain a more balanced classification scheme is using quantiles. This, by definition, assigns the same amount of values to each bin: the entire series is laid out in order and break points are assigned in a way that leaves exactly the same amount of observations between each of them. This "observation-based" approach contrasts with the "value-based" method of equal intervals and, although it can obscure the magnitude of extreme values, it can be more informative in cases with skewed distributions.
# 
# Calculating a quantiles classification with `PySAL` can be done with the following line of code:

# In[38]:


classi = mapclassify.Quantiles(imd['imd_rank'], k=7)
classi


# And, similarly, the bins can also be inspected:

# In[39]:


classi.bins


# The visualization of the distribution can be generated in a similar way as well:

# In[40]:


# Set up the figure
f, ax = plt.subplots(1)
# Plot the kernel density estimation (KDE)
sns.kdeplot(imd['imd_rank'], shade=True)
# Add a blue tick for every value at the bottom of the plot (rugs)
sns.rugplot(imd['imd_rank'], alpha=0.5)
# Loop over each break point and plot a vertical red line
for cut in classi.bins:
    plt.axvline(cut, color='red', linewidth=0.75)
# Display image
plt.show()


# And the choropleth also follows a similar pattern, with the difference that we are now using the scheme "quantiles", instead of "equal interval":

# In[41]:


imd.plot(column='imd_rank', scheme='QUANTILES', alpha=1, k=7,          cmap=plt.cm.Blues_r, 
         edgecolor='w', linewidth=0.1)


# Note how, in this case, the amount of polygons in each color is by definition much more balanced (almost equal in fact, except for rounding differences). This obscures outlier values, which get blurred by significantly smaller values in the same group, but allows to get more detail in the "most populated" part of the distribution, where instead of only white polygons, we can now discern more variability.

# ---
# 
# **[Optional exercise]**
# 
# Create a quantile kde plot and map of the actual score of the IMD (`imd_score`). 
# 
# As a bonus, make a map with 50% of transparency and no boundary lines.
# 
# ---

# ### Fisher-Jenks

# Equal interval and quantiles are only two examples of very many classification schemes to encode values into colors. Although not all of them are integrated into `geopandas`, `PySAL` includes several other classification schemes (for a detailed list, have a look at this [link](http://pysal.readthedocs.org/en/latest/library/esda/mapclassify.html)). As an example of a more sophisticated one, let us create a Fisher-Jenks choropleth:

# In[43]:


classi = mapclassify.FisherJenks(imd['imd_rank'], k=7)
classi


# This methodology aims at minimizing the variance *within* each bin while maximizing that *between* different classes.

# In[44]:


classi.bins


# Graphically, we can see how the break points are not equally spaced but are adapting to obtain an optimal grouping of observations:

# In[45]:


# Set up the figure
f, ax = plt.subplots(1)
# Plot the kernel density estimation (KDE)
sns.kdeplot(imd['imd_rank'], shade=True)
# Add a blue tick for every value at the bottom of the plot (rugs)
sns.rugplot(imd['imd_rank'], alpha=0.5)
# Loop over each break point and plot a vertical red line
for cut in classi.bins:
    plt.axvline(cut, color='red', linewidth=0.75)
# Display image
plt.show()


# Technically, however, the way to create a Fisher-Jenks map is exactly the same as before:

# In[46]:


imd.plot(column='imd_rank', scheme='fisher_jenks', 
         alpha=1, k=7, cmap=plt.cm.Blues_r, 
         edgecolor='w', linewidth=0.1)


# ## Raster basemaps

# This section requires the additional library `rasterio`:

# In[47]:


import rasterio


# Since choropleths tend to be based on administrative boundaries which do not necessarily reflect correctly the topography of a region, it may be of interest to provide a choropleth with certain geographical context. If data are available, one way to deliver this is by plotting a base raster map underneath the choropleth and allowing some transparency on the upper layer.
# 
# To do this in Python, we can combine the plotting of a raster image with the generation of a choropleth as we have seen above. First, we need to read the raster in:

# In[48]:


# Open the raster file
src = rasterio.open(liv_path)
# Extract the bounds
left, bottom, right, top = src.bounds


# At this point we are ready to generate the figure with both layers:

# In[50]:


# NOTE: this may take a little bit to run depending on your machine

# Set up the figure
f, ax = plt.subplots(1, figsize=(9, 9))
# Add raster layer
ax.imshow(src.read(1), cmap='gray', extent=(left, right, bottom, top))
# Create the choropleth
imd.plot(column='imd_score', cmap='Purples', 
         linewidth=0.1, alpha=0.75, ax=ax)
# Style the labels for the ticks
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)
# Keep axes proportionate
#plt.axis('equal')
# Display
plt.show()


# Note how the way the raster is added to the axis is different that the way we attach a vector map: the raster gets plotted through `imshow` (image show), which is a function embedded in the axis object (`ax`), while the vector object is appended by passing the axis (`ax`) through the plotting method itself.

# ## Zooming into the map

# A general map of an entire region, or urban area, can sometimes obscure particularly local patterns because they happen at a much smaller scale that cannot be perceived in the global view. One way to solve this is by providing a focus of a smaller part of the map in a separate figure. Although there are many ways to do this in Python, the most straightforward one is to reset the limits of the axes to center them in the area of interest.
# 
# As an example, let us consider the quantile map produced above:

# In[51]:


imd.plot(column='imd_rank', scheme='QUANTILES', 
         alpha=1, k=7, cmap=plt.cm.Blues_r, 
         edgecolor='w', linewidth=0.1)


# If we want to focus on the city centre, say the area of the map more or less between coordinates 387,000 and 391,000 on the vertical axis, and 332,000 and 337,000 on the horizontal one, creating the figure involves the following:

# In[52]:


# Setup the figure
f, ax = plt.subplots(1)
# Set background color of the axis
ax.set_facecolor('#D5E3D8')
# Draw the choropleth
imd.plot(column='imd_rank', scheme='QUANTILES', k=7,          cmap=plt.cm.Purples_r, ax=ax)
# [Optional] Keep axes proportionate
plt.axis('equal')
# Redimensionate X and Y axes to desired bounds
ax.set_ylim(387000, 391000)
ax.set_xlim(332000, 337000)
# Show image
plt.show()


# Note how, if we decide to keep the axes proportionate, it needs to be done *before* resetting the limits, as otherwise the change will not have an effect.

# ---
# 
# <a rel="repo" href="https://github.com/darribas/gds19"><img alt="@darribas/gds19" style="border-width:0" src="../../GitHub-Mark.png" /></a>
# 
# This notebook, as well as the entire set of materials, code, and data included
# in this course are available as an open Github repository available at: [`https://github.com/darribas/gds19`](https://github.com/darribas/gds19)
# 
# <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">Geographic Data Science'19</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="http://darribas.org" property="cc:attributionName" rel="cc:attributionURL">Dani Arribas-Bel</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
# 
