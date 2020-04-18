#!/usr/bin/env python
# coding: utf-8

# # Spatial weights

# In this session we will be learning the ins and outs of one of the key pieces in spatial analysis: spatial weights matrices. These are structured sets of numbers that formalize geographical relationships between the observations in a dataset. Essentially, a spatial weights matrix of a given geography is a positive definite matrix of dimensions $N$ by $N$, where $N$ is the total number of observations:
# 
# $$
# W = \left(\begin{array}{cccc}
# 0 & w_{12} & \dots & w_{1N} \\
# w_{21} & \ddots & w_{ij} & \vdots \\
# \vdots & w_{ji} & 0 & \vdots \\
# w_{N1} & \dots & \dots & 0 
# \end{array} \right)
# $$
# 
# where each cell $w_{ij}$ contains a value that represents the degree of spatial contact or interaction between observations $i$ and $j$. A fundamental concept in this context is that of *neighbor* and *neighborhood*. By convention, elements in the diagonal ($w_{ij}$) are set to zero. A *neighbor* of a given observation $i$ is another observation with which $i$ has some degree of connection. In terms of $W$, $i$ and $j$ are thus neighbors if $w_{ij} > 0$. Following this logic, the neighborhood of $i$ will be the set of observations in the system with which it has certain connection, or those observations with a weight greater than zero.
# 
# There are several ways to create such matrices, and many more to transform them so they contain an accurate representation that aligns with the way we understand spatial interactions between the elements of a system. In this session, we will introduce the most commonly used ones and will show how to compute them with `PySAL`.

# In[3]:


# get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
import pandas as pd
from pysal.lib import weights
from pysal.lib.io import open as psopen
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import get_backend


# In[4]:


get_backend()


# ## Data
# 
# For this tutorial, we will use again the recently released 2015 Index of Multiple Deprivation (IMD) for England and Wales. This dataset can be most easily downloaded from the CDRC data store ([link](https://data.cdrc.ac.uk/dataset/cdrc-english-indices-of-deprivation-2015-geodata-pack-liverpool-e08000012)) and, since it already comes both in tabular as well as spatial data format (shapefile), it does not need merging or joining to additional geometries.
# 
# In addition, we will be using the lookup between LSOAs and Medium Super Output Areas (MSOAs), which can be downloaded on this [link](http://www.ons.gov.uk/ons/external-links/social-media/g-m/2011-oas-to-2011-lower-layer-super-output-areas--lsoas---middle-layer-super-output-areas--msoa--and-lads.html). This connects each LSOA polygon to the MSOA they belong to. MSOAs are a coarser geographic delineation from the Office of National Statistics (ONS), within which LSOAs are nested. That is, no LSOA boundary crosses any of an MSOA.
# 
# As usual, let us set the paths to the folders containing the files before anything so we can then focus on data analysis exclusively (keep in mind the specific paths will probably be different for your computer):

# In[5]:


# This will be different on your computer and will depend on where
# you have downloaded the files
imd_shp = '../data/E08000012/shapefiles/E08000012.shp'
lookup_path = '../data/output_areas_(2011)_to_lower_layer_super_output_areas_(2011)_to_middle_layer_super_output_areas_(2011)_to_local_authority_districts_(2011)_e+w_lookup/'


# Let us load the IMD data first:

# In[6]:


# Read the file in
imd = gpd.read_file(imd_shp)
# Index it on the LSOA ID
imd = imd.set_index('LSOA11CD', drop=False)
# Display summary
imd.info()


# ## Building spatial weights in `PySAL`

# In[8]:


imd.plot(column='imd_rank')
plt.show()


# ### Contiguity
# 
# Contiguity weights matrices define spatial connections through the existence of common boundaries. This makes it directly suitable to use with polygons: if two polygons share boundaries to some degree, they will be labeled as neighbors under these kinds of weights. Exactly how much they need to share is what differenciates the two approaches we will learn: queen and rook.

# * **Queen**
# 
# Under the queen criteria, two observations only need to share a vortex (a single point) of their boundaries to be considered neighbors. Constructing a weights matrix under these principles can be done by running:

# In[9]:


w_queen = weights.Queen.from_dataframe(imd, idVariable="LSOA11CD")
w_queen


# The command above creates an object `w_queen` of the class `W`. This is the format in which spatial weights matrices are stored in `PySAL`. By default, the weights builder (`Queen.from_dataframe`) will use the index of the table, which is useful so we can keep everything in line easily.
# 
# A `W` object can be queried to find out about the contiguity relations it contains. For example, if we would like to know who is a neighbor of observation `E01006690`:

# In[10]:


w_queen['E01006690']


# This returns a Python dictionary that contains the ID codes of each neighbor as keys, and the weights they are assigned as values. Since we are looking at a raw queen contiguity matrix, every neighbor gets a weight of one. If we want to access the weight of a specific neighbor, `E01006691` for example, we can do recursive querying:

# In[11]:


w_queen['E01006690']['E01006691']


# `W` objects also have a direct way to provide a list of all the neighbors or their weights for a given observation. This is thanks to the `neighbors` and `weights` attributes:

# In[12]:


w_queen.neighbors['E01006690']


# In[13]:


w_queen.weights['E01006690']


# Once created, `W` objects can provide much information about the matrix, beyond the basic attributes one would expect. We have direct access to the number of neighbors each observation has via the attribute `cardinalities`. For example, to find out how many neighbors observation `E01006524` has:

# In[14]:


w_queen.cardinalities['E01006524']


# Since `cardinalities` is a dictionary, it is direct to convert it into a `Series` object:

# In[15]:


queen_card = pd.Series(w_queen.cardinalities)
queen_card.head()


# This allows, for example, to access quick plotting, which comes in very handy to get an overview of the size of neighborhoods in general:

# In[29]:


f, ax = plt.subplots(1)
sns.distplot(queen_card, bins=10, ax=ax)
plt.title('a')
plt.show()


# The figure above shows how most observations have around five neighbors, but there is some variation around it. The distribution also seems to follow a symmetric form, where deviations from the average occur both in higher and lower values almost evenly.
# 
# Some additional information about the spatial relationships contained in the matrix are also easily available from a `W` object. Let us tour over some of them:

# In[16]:


# Number of observations
w_queen.n


# In[17]:


# Average number of neighbors
w_queen.mean_neighbors


# In[18]:


# Min number of neighbors
w_queen.min_neighbors


# In[19]:


# Max number of neighbors
w_queen.max_neighbors


# In[20]:


# Islands (observations disconnected)
w_queen.islands


# In[21]:


# Order of IDs (first five only in this case)
w_queen.id_order[:5]


# Spatial weight matrices can be explored visually in other ways. For example, we can pick an observation and visualize it in the context of its neighborhood. The following plot does exactly that by zooming into the surroundings of LSOA `E01006690` and displaying its polygon as well as those of its neighbors:

# In[23]:


# Setup figure
f, ax = plt.subplots(1, figsize=(6, 6))
# Plot base layer of polygons
imd.plot(ax=ax, facecolor='k', linewidth=0.1)
# Select focal polygon
# NOTE we pass both the area code and the column name
#      (`geometry`) within brackets!!!
focus = imd.loc[['E01006690'], ['geometry']]
# Plot focal polygon
focus.plot(facecolor='red', alpha=1, linewidth=0, ax=ax)
# Plot neighbors
neis = imd.loc[w_queen['E01006690'], :]
neis.plot(ax=ax, facecolor='lime', linewidth=0)
# Title
plt.suptitle("Queen neighbors of `E01006690`")
# Style and display on screen
ax.set_ylim(388000, 393500)
ax.set_xlim(336000, 339500)
plt.show()


# Note how the figure is built gradually, from the base map (L. 4-5), to the focal point (L. 9), to its neighborhood (L. 11-13). Once the entire figure is plotted, we zoom into the area of interest (L. 19-20).

# * **Rook**
# 
# Rook contiguity is similar to and, in many ways, superseded by queen contiguity. However, since it sometimes comes up in the literature, it is useful to know about it. The main idea is the same: two observations are neighbors if they share some of their boundary lines. However, in the rook case, it is not enough with sharing only one point, it needs to be at least a segment of their boundary. In most applied cases, these differences usually boil down to how the geocoding was done, but in some cases, such as when we use raster data or grids, this approach can differ more substantively and it thus makes more sense.
# 
# From a technical point of view, constructing a rook matrix is very similar:

# In[24]:


w_rook = weights.Rook.from_dataframe(imd)
w_rook


# The output is of the same type as before, a `W` object that can be queried and used in very much the same way as any other one.

# ---
# 
# **[Optional exercise]**
# 
# Create a similar map for the rook neighbors of polygon `E01006580`. 
# 
# How would it differ if the spatial weights were created based on the queen criterion?
# 
# ---

# ### Distance
# 
# Distance based matrices assign the weight to each pair of observations as a function of how far from each other they are. How this is translated into an actual weight varies across types and variants, but they all share that the ultimate reason why two observations are assigned some weight is due to the distance between them.

# * **K-Nearest Neighbors**
# 
# One approach to define weights is to take the distances between a given observation and the rest of the set, rank them, and consider as neighbors the $k$ closest ones. That is exactly what the $k$-nearest neighbors (KNN) criterium does.
# 
# To calculate KNN weights, we can use a similar function as before and derive them from a shapefile:

# In[25]:


knn5 = weights.KNN.from_dataframe(imd, k=5)
knn5


# Note how we need to specify the number of nearest neighbors we want to consider with the argument `k`. Since it is a polygon shapefile that we are passing, the function will automatically compute the centroids to derive distances between observations. Alternatively, we can provide the points in the form of an array, skipping this way the dependency of a file on disk:

# In[26]:


# Extract centroids
cents = imd.centroid
# Extract coordinates into an array
pts = np.array([(pt.x, pt.y) for pt in cents])
# Compute KNN weights
knn5_from_pts = weights.KNN.from_array(pts, k=5)
knn5_from_pts


# * **Distance band**
#  
# Another approach to build distance-based spatial weights matrices is to draw a circle of certain radious and consider neighbor every observation that falls within the circle. The technique has two main variations: binary and continuous. In the former one, every neighbor is given a weight of one, while in the second one, the weights can be further tweaked by the distance to the observation of interest.
# 
# To compute binary distance matrices in `PySAL`, we can use the following command:

# In[27]:


w_dist1kmB = weights.DistanceBand.from_dataframe(imd, 1000)


# **NOTE** how we approach this in a different way, by using the method `from_shapefile` we do not build the `W` based on the table `imd`, but instead use directly the file the table came from (which we point at using `imd_shp`, the path). Note also how we need to include the name of the column where the index of the table is stored (`LSOA11CD`, the LSOA code) so the matrix is aligned and indexed in the same way as the tabl. Once built, however, the output is of the same kind as before, a `W` object.
# 
# This creates a binary matrix that considers neighbors of an observation every polygon whose centroid is closer than 1,000 metres (1Km) of the centroid of such observation. Check, for example, the neighborhood of polygon `E01006690`:

# In[28]:


w_dist1kmB['E01006690']


# Note that the units in which you specify the distance directly depend on the CRS in which the spatial data are projected, and this has nothing to do with the weights building but it can affect it significantly. Recall how you can check the CRS of a `GeoDataFrame`:

# In[29]:


imd.crs


# In this case, you can see the unit is expressed in metres (`m`), hence we set the threshold to 1,000 for a circle of 1km of radious.
# 
# An extension of the weights above is to introduce further detail by assigning different weights to different neighbors within the radious circle based on how far they are from the observation of interest. For example, we could think of assigning the inverse of the distance between observations $i$ and $j$ as $w_{ij}$. This can be computed with the following command:

# In[31]:


w_dist1kmC = weights.DistanceBand.from_dataframe(imd, 1000, binary=False)


# In `w_dist1kmC`, every observation within the 1km circle is assigned a weight equal to the inverse distance between the two:
# 
# $$
# w_{ij} = \dfrac{1}{d_{ij}}
# $$
# 
# This way, the further apart $i$ and $j$ are from each other, the smaller the weight $w_{ij}$ will be.
# 
# Contrast the binary neighborhood with the continuous one for `E01006690`:

# In[32]:


w_dist1kmC['E01006690']


# ---
# 
# **[Optional exercise]**
# 
# Explore the help for functions `weights.DistanceBand.from_array` and try to use them to replicate `w_dist1kmB` and `w_dist1kmC`.
# 
# ---

# Following this logic of more detailed weights through distance, there is a temptation to take it further and consider everyone else in the dataset as a neighbor whose weight will then get modulated by the distance effect shown above. However, although conceptually correct, this approach is not always the most computationally or practical one. Because of the nature of spatial weights matrices, particularly because of the fact their size is $N$ by $N$, they can grow substantially large. A way to cope with this problem is by making sure they remain fairly *sparse* (with many zeros). Sparsity is typically ensured in the case of contiguity or KNN by construction but, with inverse distance, it needs to be imposed as, otherwise, the matrix could be potentially entirely dense (no zero values other than the diagonal). In practical terms, what is usually done is to impose a distance threshold beyond which no weight is assigned and interaction is assumed to be non-existent. Beyond being computationally feasible and scalable, results from this approach usually do not differ much from a fully "dense" one as the additional information that is included from further observations is almost ignored due to the small weight they receive. In this context, a commonly used threshold, although not always best, is that which makes every observation to have at least one neighbor. 
# 
# Such a threshold can be calculated as follows:

# In[33]:


min_thr = weights.min_threshold_dist_from_shapefile(imd_shp.replace(".gpkg", ".shp"))
min_thr


# Which can then be used to calculate an inverse distance weights matrix:

# In[34]:


w_min_dist = weights.DistanceBand.from_dataframe(imd, min_thr, binary=False)


# ---
# 
# **[Optional extension. Lecture figure]**
# 
# Below is how to build a visualization for distance-based weights that displays the polygons, highlighting the focus and its neighbors, and then overlays the centroids and the buffer used to decide whether a polygon is a neighbor or not. Since this is distance-based weights, there needs to be a way to establish distance between two polygons and, in this case, the distance between their centroids is used.

# In[35]:


# Setup figure
f, ax = plt.subplots(1, figsize=(4, 4))
# Plot base layer of polygons
imd.plot(ax=ax, facecolor='k', linewidth=0.1)
# Select focal polygon
# NOTE we pass both the area code and the column name
#      (`geometry`) within brackets!!!
focus = imd.loc[['E01006690'], ['geometry']]
# Plot focal polygon
focus.plot(facecolor='red', alpha=1, linewidth=0, ax=ax)
# Plot neighbors
neis = imd.loc[w_dist1kmC['E01006690'], :]
neis.plot(ax=ax, facecolor='lime', linewidth=0)
# Plot 1km buffer
buf = focus.centroid.buffer(1000)
buf.plot(edgecolor='red', facecolor='none', ax=ax)
# Plot centroids of neighbor
pts = np.array([(pt.x, pt.y) for pt in imd.centroid])
ax.plot(pts[:, 0], pts[:, 1], color='#00d8ea', 
        linewidth=0, alpha=0.75, marker='o', markersize=4)
# Title
f.suptitle("Neighbors within 1km of `E01006690`")
# Style, zoom and display on screen
ax.set_ylim(388000, 393500)
ax.set_xlim(336000, 339500)
plt.show()


# ---

# ### Block weights

# Block weights connect every observation in a dataset that belongs to the same category in a list provided ex-ante. Usually, this list will have some relation to geography an the location of the observations but, technically speaking, all one needs to create block weights is a list of memberships. In this class of weights, neighboring observations, those in the same group, are assigned a weight of one, and the rest receive a weight of zero.
# 
# In this example, we will build a spatial weights matrix that connects every LSOA with all the other ones in the same MSOA. To do this, we first need a lookup list that connects both kinds of geographies:

# In[36]:


# NOTE: disregard the warning in pink that might come from running 
#       this cell
file_name = 'OA11_LSOA11_MSOA11_LAD11_EW_LUv2.csv'
lookup = pd.read_csv(lookup_path+file_name, encoding='iso-8859-1')
lookup = lookup[['LSOA11CD', 'MSOA11CD']].drop_duplicates(keep='last')                                         .set_index('LSOA11CD')['MSOA11CD']
lookup.head()


# Since the original file contains much more information than we need for this exercise, note how in line 2 we limit the columns we keep to only two, `LSOA11CD` and `MSOA11CD`. We also add an additional command, `drop_duplicates`, which removes elements whose index is repeated more than once, as is the case in this dataset (every LSOA has more than one row in this table). By adding the `take_last` argument, we make sure that one and only one element of each index value is retained. For ease of use later on, we set the index, that is the name of the rows, to `LSOA11CD`. This will allow us to perform efficient lookups without having to perform full `DataFrame` queries, and it is also a more computationally efficient way to select observations.
# 
# For example, if we want to know in which MSOA the polygon `E01000003` is, we just need to type:

# In[37]:


lookup.loc['E01000003']


# With the lookup in hand, let us append it to the IMD table to keep all the necessary pieces in one place only:

# In[38]:


imd['MSOA11CD'] = lookup


# Now we are ready to build a block spatial weights matrix that connects as neighbors all the LSOAs in the same MSOA. Using `PySAL`, this is a one-line task:

# In[39]:


w_block = weights.block_weights(imd['MSOA11CD'])


# In this case, `PySAL` does not allow to pass the argument `idVariable` as above. As a result, observations are named after the the order the occupy in the list:

# In[40]:


w_block[0]


# The first element is neighbor of observations 218, 129, 220, and 292, all of them with an assigned weight of 1. However, it is possible to correct this by using the additional method `remap_ids`:

# In[41]:


w_block.remap_ids(imd.index)


# Now if you try `w_bloc[0]`, it will return an error. But if you query for the neighbors of an observation by its LSOA id, it will work:

# In[42]:


w_block['E01006512']


# ---
# 
# **[Optional exercise]**
# 
# For block weights, create a similar map to that of queen neighbors of polygon `E01006690`.
# 
# ---

# ## Standardizing `W` matrices
# 
# In the context of many spatial analysis techniques, a spatial weights matrix with raw values (e.g. ones and zeros for the binary case) is not always the best suiting one for analysis and some sort of transformation is required. This implies modifying each weight so they conform to certain rules. `PySAL` has transformations baked right into the `W` object, so it is possible to check the state of an object as well as to modify it.
# 
# Consider the original queen weights, for observation `E01006690`:

# In[43]:


w_queen['E01006690']


# Since it is contiguity, every neighbor gets one, the rest zero weight. We can check if the object `w_queen` has been transformed or not by calling the argument `transform`:

# In[44]:


w_queen.transform


# where `O` stands for "original", so no transformations have been applied yet. If we want to apply a row-based transformation, so every row of the matrix sums up to one, we modify the `transform` attribute as follows:

# In[45]:


w_queen.transform = 'R'


# Now we can check the weights of the same observation as above and find they have been modified:

# In[46]:


w_queen['E01006690']


# Save for precission issues, the sum of weights for all the neighbors is one:

# In[50]:


pd.Series(w_queen['E01006690']).sum()


# Returning the object back to its original state involves assigning `transform` back to original:

# In[51]:


w_queen.transform = 'O'


# In[52]:


w_queen['E01006690']


# `PySAL` supports the following transformations:
# 
# * `O`: original, returning the object to the initial state.
# * `B`: binary, with every neighbor having assigned a weight of one.
# * `R`: row, with all the neighbors of a given observation adding up to one.
# * `V`: variance stabilizing, with the sum of all the weights being constrained to the number of observations.

# ## Reading and Writing spatial weights in `PySAL`
# 
# Sometimes, if a dataset is very detailed or large, it can be costly to build the spatial weights matrix of a given geography and, despite the optimizations in the `PySAL` code, the computation time can quickly grow out of hand. In these contexts, it is useful to not have to re-build a matrix from scratch every time we need to re-run the analysis. A useful solution in this case is to build the matrix once, and save it to a file where it can be reloaded at a later stage if needed.
# 
# `PySAL` has a common way to write any kind of `W` object into a file using the command `open`. The only element we need to decide for ourselves beforehand is the format of the file. Although there are several formats in which spatial weight matrices can be stored (have a look at the [list](http://pysal.readthedocs.org/en/latest/users/tutorials/fileio.html) of supported ones by `PySAL`), we will focused on the two most commonly used ones:

# * **`.gal`** files for contiguity weights
# 
# Contiguity spatial weights can be saved into a `.gal` file with the following commands:

# In[56]:


weights_dir = '../data/weights/'


# In[57]:


# Open file to write into
fo = psopen(weights_dir + 'imd_queen.gal', 'w')
# Write the matrix into the file
fo.write(w_queen)
# Close the file
fo.close()


# The process is composed by the following three steps:
# 
# 1. Open a target file for `w`riting the matrix, hence the `w` argument. In this case, if a file `imd_queen.gal` already exists, it will be overwritten, so be careful.
# 1. Write the `W` object into the file.
# 1. Close the file. This is important as some additional information is written into the file at this stage, so failing to close the file might have unintended consequences.
# 
# Once we have the file written, it is possible to read it back into memory with the following command:

# In[58]:


w_queen2 = psopen(weights_dir + 'imd_queen.gal', 'r').read()
w_queen2


# Note how we now use `r` instead of `w` because we are `r`eading the file, and also notice how we open the file and, in the same line, we call `read()` directly.

# * **`.gwt`** files for distance-based weights.
# 
# A very similar process to the one above can be used to read and write distance based weights. The only difference is specifying the right file format, `.gwt` in this case. So, if we want to write `w_dist1km` into a file, we will run:

# In[59]:


# Open file
fo = psopen(weights_dir + 'imd_dist1km.gwt', 'w')
# Write matrix into the file
fo.write(w_dist1kmC)
# Close file
fo.close()


# And if we want to read the file back in, all we need to do is:

# In[60]:


w_dist1km2 = psopen(weights_dir + 'imd_dist1km.gwt', 'r').read()


# Note how, in this case, you will probably receive a warning alerting you that there was not a `DBF` relating to the file. This is because, by default, `PySAL` takes the order of the observations in a `.gwt` from a shapefile. If this is not provided, `PySAL` cannot entirely determine all the elements and hence the resulting `W` might not be complete (islands, for example, can be missing). To fully complete the reading of the file, we can remap the ids as we have seen above:

# In[61]:


w_dist1km2.remap_ids(imd.index)


# ## Spatial Lag
# 
# One of the most direct applications of spatial weight matrices is the so-called *spatial lag*. The spatial lag of a given variable is the product of a spatial weight matrix and the variable itself:
# 
# $$
# Y_{sl} = W Y
# $$
# 
# where $Y$ is a Nx1 vector with the values of the variable. Recall that the product of a matrix and a vector equals the sum of a row by column element multiplication for the resulting value of a given row. In terms of the spatial lag:
# 
# $$
# y_{sl-i} = \displaystyle \sum_j w_{ij} y_j
# $$
# 
# If we are using row-standardized weights, $w_{ij}$ becomes a proportion between zero and one, and $y_{sl-i}$ can be seen as the average value of $Y$ in the neighborhood of $i$.
# 
# The spatial lag is a key element of many spatial analysis techniques, as we will see later on and, as such, it is fully supported in `PySAL`. To compute the spatial lag of a given variable, `imd_score` for example:

# In[62]:


# Row-standardize the queen matrix
w_queen.transform = 'R'
# Compute spatial lag of `imd_score`
w_queen_score = weights.lag_spatial(w_queen, imd['imd_score'])
# Print the first five elements
w_queen_score[:5]


# Line 4 contains the actual computation, which is highly optimized in `PySAL`. Note that, despite passing in a `pd.Series` object, the output is a `numpy` array. This however, can be added directly to the table `imd`:

# In[63]:


imd['w_queen_score'] = w_queen_score


# ---
# 
# **[Optional exercise]**
# 
# Explore the spatial lag of `w_queen_score` by constructing a density/histogram plot similar to those created in Lab 2. Compare these with one for `imd_score`. What differences can you tell?
# 
# <!--
# sns.distplot(imd['imd_score'])
# 
# sns.distplot(imd['w_queen_score'])
# -->
# 
# ---

# In[65]:


# sns.distplot(imd['imd_score'])

sns.distplot(imd['w_queen_score'])


# ## Moran Plot
# 
# The Moran Plot is a graphical way to start exploring the concept of spatial autocorrelation, and it is a good application of spatial weight matrices and the spatial lag. In essence, it is a standard scatter plot in which a given variable (`imd_score`, for example) is plotted against *its own* spatial lag. Usually, a fitted line is added to include more information:

# In[67]:


# Setup the figure and axis
f, ax = plt.subplots(1, figsize=(9, 9))
# Plot values
sns.regplot(x="imd_score", y="w_queen_score", data=imd, ci=None)
# Display
plt.show()


# In order to easily compare different scatter plots and spot outlier observations, it is common practice to standardize the values of the variable before computing its spatial lag and plotting it. This can be accomplished by substracting the average value and dividing the result by the standard deviation:
# 
# $$
# z_i = \dfrac{y - \bar{y}}{\sigma_y}
# $$
# 
# where $z_i$ is the standardized version of $y_i$, $\bar{y}$ is the average of the variable, and $\sigma$ its standard deviation.
# 
# Creating a standardized Moran Plot implies that average values are centered in the plot (as they are zero when standardized) and dispersion is expressed in standard deviations, with the rule of thumb of values greater or smaller than two standard deviations being *outliers*. A standardized Moran Plot also partitions the space into four quadrants that represent different situations:
# 
# 1. High-High (*HH*): values above average surrounded by values above average.
# 1. Low-Low (*LL*): values below average surrounded by values below average.
# 1. High-Low (*HL*): values above average surrounded by values below average.
# 1. Low-High (*LH*): values below average surrounded by values above average.
# 
# These will be further explored once spatial autocorrelation has been properly introduced in subsequent lectures.

# In[68]:


# Standardize the IMD scores
std_imd = (imd['imd_score'] - imd['imd_score'].mean()) / imd['imd_score'].std()
# Compute the spatial lag of the standardized version and save is as a 
# Series indexed as the original variable
std_w_imd = pd.Series(weights.lag_spatial(w_queen, std_imd), index=std_imd.index)
# Setup the figure and axis
f, ax = plt.subplots(1, figsize=(9, 9))
# Plot values
sns.regplot(x=std_imd, y=std_w_imd, ci=None)
# Add vertical and horizontal lines
plt.axvline(0, c='k', alpha=0.5)
plt.axhline(0, c='k', alpha=0.5)
# Display
plt.show()


# ---
# 
# **[Optional exercise]**
# 
# Create a standardized Moran Plot for each of the components of the IMD:
# 
# * Crime
# * Education
# * Employment
# * Health
# * Housing
# * Income
# * Living environment
# 
# **Bonus** if you can generate all the plots with a `for` loop.
# 
# **Bonus-II** if you explore the functionality of Seaborn's `jointplot` ([link](http://stanford.edu/~mwaskom/software/seaborn/tutorial/regression.html#plotting-a-regression-in-other-contexts) and [link](http://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.jointplot.html#seaborn.jointplot)) to create a richer Moran plot.
# 
# <!--
# w_queen.transform = 'R'
# for col in ['crime', 'education', 'employment', 'health', 'housing', 'income', 'living_env']:
#     stdd = (imd[col] - imd[col].mean()) / imd[col].std()
#     sl = pd.Series(ps.lag_spatial(w_queen, stdd), index=stdd.index)
#     sns.jointplot(x=stdd, y=sl, kind="reg")
# -->
# 
# 

# In[70]:


w_queen.transform = 'R'
for col in ['crime', 'education', 'employment', 'health', 'housing', 'income', 'living_env']:
    stdd = (imd[col] - imd[col].mean()) / imd[col].std()
    sl = pd.Series(weights.lag_spatial(w_queen, stdd), index=stdd.index)
    sns.jointplot(x=stdd, y=sl, kind="reg")


# ---
# 
# <a rel="repo" href="https://github.com/darribas/gds19"><img alt="@darribas/gds19" style="border-width:0" src="../../GitHub-Mark.png" /></a>
# 
# This notebook, as well as the entire set of materials, code, and data included
# in this course are available as an open Github repository available at: [`https://github.com/darribas/gds19`](https://github.com/darribas/gds19)
# 
# <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">Geographic Data Science'19</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="http://darribas.org" property="cc:attributionName" rel="cc:attributionURL">Dani Arribas-Bel</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
