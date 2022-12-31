# Required function for running this assignment
# Written by Mehdi Rezvandehy

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib import gridspec
import matplotlib.patheffects as pe
import scipy.linalg 
import scipy.stats as ss
from scipy.stats import norm
import utm
from scipy.stats import reciprocal, uniform
from scipy.stats import randint
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from collections import Counter
from sklearn.datasets import make_classification
import matplotlib.ticker as mtick
from sklearn.metrics import precision_recall_curve
from tensorflow.keras import metrics
from sklearn.ensemble import VotingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import pickle
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')
##########################################################################################################

def loc_prob_plot(x_1, y_1,prob,title,alpha1,s,save,vmin,vmax,ang):
    """ Function to plot location map and pie chart of """
    font = {'size'   : 7.5}
    mpl.rc('font', **font)
    
    fig, axs = plt.subplots(figsize=(6,5), dpi= 200, facecolor='w', edgecolor='k')
    Name=["" for x in range(6)]
    Name[0:6]='Calgary','Fort McMurray','Edmonton','Rainbow',\
     'Grande Prairie','Lloydminster'
    
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.3, 1], wspace=0.28)
    
    ax1=plt.subplot(gs[0]) 
    ax1.spines['left'].set_visible(True)
    ax1.spines['bottom'].set_visible(True)  
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)  
    with open('./Data/Location names.txt') as f:
        xn = []
        yn = [] 
        for line in f:
            p = line.split()
            tmp=utm.from_latlon(float(p[2]), float(p[1]),force_zone_number=12)
            xn.append((float(tmp[0])-(-80563.0))/1000)
            yn.append((float(tmp[1]-5428042.5))/1000)
    plt.plot(xn,yn,'*y',markersize=7,linewidth=1)        
    for i in range(5):
        plt.text(xn[i], (yn[i]-18), Name[i],fontsize=5.5,color='y', fontweight='bold',style='oblique', ha='center',
         va='top', wrap=True,bbox=dict(facecolor='k', alpha=1,pad=0))      
        
    # Clip to Alberta province
    with open('./Data/Alberta.dat') as f:
        path = []
        Name = []   
        for line in f:
            p = line.split()
            Name.append(p[0])
            tmp=utm.from_latlon(float(p[1]), float(p[0]),force_zone_number=12)
            aa=[tmp[0]/1000-(-80563.0/1000),tmp[1]/1000-(5428042.5/1000)]
            path.append(aa)
    path1 = np.asarray(path)
    path = Path(path1)
    patch = PathPatch(path, facecolor='none',linewidth=0.8)
    plt.gca().add_patch(patch)
    
    #Serious
    plt.scatter(x_1, y_1,c=prob,s=s, marker='o',cmap='jet',vmin=vmin,vmax=vmax,alpha=alpha1,edgecolors='k')  
    plt.colorbar(shrink=0.5,label='Probability of Serious Leakage',orientation='vertical')

    plt.title(title,fontsize=8.5)    
    plt.xlabel('Easting (km)',fontsize='9.5')
    plt.ylabel('Northing (km)',fontsize='9.5')
    plt.gca().set_aspect('0.9')
    plt.xlim((-30, 660)) 
    plt.ylim((-30, 1270))
    
    ax2=plt.subplot(gs[1]) 
    
    y_pred=[1 if i>0.5 else 0 for i in prob]
    # Data to plot
    labels = 'Serious: \n'+str(sum(y_pred))+' Wells', 'Non-serious: \n'+\
    str(len(y_pred)-sum(y_pred))+' Wells'
    sizes = [int(sum(y_pred)), int(len(y_pred)-sum(y_pred))]
    colors = ['red', 'blue']
    explode = (0.1, 0)  # explode 1st slice
    
    # Plot
    _, _, autopcts = ax2.pie(sizes,explode= explode, labels=labels, autopct='%1.1f%%',
        shadow=True,startangle=ang, colors=colors)
    plt.setp(autopcts, **{'color':'white', 'weight':'bold', 'fontsize':9})
    
    plt.title(' Number and Percentage \n of each Class',y=0.69,fontsize=8)  
    plt.axis('equal')    
    plt.show()
    fig.tight_layout(w_pad=1.2)   
    if(save): fig.savefig(save, dpi=500, bbox_inches='tight')
    plt.show()
    
##########################################################################################
def loc_pie_plot(x_1, y_1, x_2, y_2, count_0, count_1,title,s1,s2,alpha1,alpha2,save,startangle=140):
    """ Function to plot location map and pie chart of """
    font = {'size'   : 7.5}
    mpl.rc('font', **font)
    
    fig, axs = plt.subplots(figsize=(6,5), dpi= 200, facecolor='w', edgecolor='k')
    Name=["" for x in range(6)]
    Name[0:6]='Calgary','Fort McMurray','Edmonton','Rainbow',\
     'Grande Prairie','Lloydminster'
    
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.1)
    
    ax1=plt.subplot(gs[0]) 
    ax1.spines['left'].set_visible(True)
    ax1.spines['bottom'].set_visible(True)  
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)  
    with open('./Data/Location names.txt') as f:
        xn = []
        yn = [] 
        for line in f:
            p = line.split()
            tmp=utm.from_latlon(float(p[2]), float(p[1]),force_zone_number=12)
            xn.append((float(tmp[0])-(-80563.0))/1000)
            yn.append((float(tmp[1]-5428042.5))/1000)
    plt.plot(xn,yn,'*y',markersize=7,linewidth=1)        
    for i in range(5):
        plt.text(xn[i], (yn[i]-18), Name[i],fontsize=7.5,color='y', fontweight='bold',style='oblique', ha='center',
         va='top', wrap=True,bbox=dict(facecolor='k', alpha=1,pad=0))      
        
    # Clip to Alberta province
    with open('./Data/Alberta.dat') as f:
        path = []
        Name = []   
        for line in f:
            p = line.split()
            Name.append(p[0])
            tmp=utm.from_latlon(float(p[1]), float(p[0]),force_zone_number=12)
            aa=[tmp[0]/1000-(-80563.0/1000),tmp[1]/1000-(5428042.5/1000)]
            path.append(aa)
    path1 = np.asarray(path)
    path = Path(path1)
    patch = PathPatch(path, facecolor='none',linewidth=0.8)
    plt.gca().add_patch(patch)
    
    #Serious
    label= 'Serious'
    plt.scatter(x_1, y_1,s=s1,c='r', marker='o',alpha=alpha1,label=label)  
    ###################################################################
    
    #Non Serious
    label= 'Non-serious'
    plt.scatter(x_2, y_2,s=s2,c='b', marker='*',alpha=alpha2,label=label) 
    ###################################################################
    
    plt.title(title,fontsize=9.5)    
    plt.xlabel('Easting (km)',fontsize='9.5')
    plt.ylabel('Northing (km)',fontsize='9.5')
    plt.gca().set_aspect('0.9')
    #plt.legend(bbox_to_anchor=(0.7,-0.1),fontsize='7')
    plt.legend(loc=9,bbox_to_anchor=(0.25, 0.12), ncol=1,fontsize='8',markerscale=1.6)
    
    plt.xlim((-30, 660))   # set the xlim to left, right
    plt.ylim(-30, 1270)     # set the xlim to left, right
    
    ax2=plt.subplot(gs[1]) 
    
    # Data to plot
    labels = 'Serious: \n'+str(count_1)+' Wells', 'Non-serious: \n'+\
    str(count_0)+' Wells'
    sizes = [count_1, count_0]
    colors = ['red', 'blue']
    explode = (0.1, 0)  # explode 1st slice
    # Plot
    
    _, _, autopcts = ax2.pie(sizes,explode= explode, labels=labels, autopct='%1.1f%%',
        shadow=True,startangle=startangle, colors=colors)
    plt.setp(autopcts, **{'color':'white', 'weight':'bold', 'fontsize':9})
    
    
#    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
#    autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title(' Number and Percentage \n of each Class',y=0.69,fontsize=8)  
    plt.axis('equal')
    plt.show()
    
    fig.tight_layout(w_pad=1.2)   
    
    if(save): fig.savefig(save, dpi=500, bbox_inches='tight')
    plt.show()    
    
###########################################################################################################    

def smoothed_mean(df_1, cat, target, weight):
    '''Function for categorical decoding'''
    # global mean
    mean = df_1[target].mean()

    # number of values and the mean of each category
    agg = df_1.groupby(cat)[target].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']

    # "smoothed" means
    smooth = (counts * means + weight * mean) / (counts + weight)

    # Replace each value by the according smoothed mean
    return df_1[cat].map(smooth), smooth.to_dict()
  
    
###########################################################################################################     

def coef(coefs,names):
    '''Function to plot correlation coefficient'''
    r_ = pd.DataFrame( { 'coef': coefs, 'positive': coefs>=0  }, index = names )
    r_ = r_.sort_values(by=['coef'])
    r_['coef'].plot(kind='barh', color=r_['positive'].map({True: 'b', False: 'r'}))    
       
###########################################################################################################  

def CrossPlot (x,y,in_,x__1,x__2,title,xlabl,ylabl,loc,xlimt,ylimt,markersize,axt=None,scale=0.8,alpha=0.1,loc_1=2):
    '''Cross plot between two variables'''
    ax1 = axt or plt.axes()
    x=np.array(x)
    y=np.array(y)    
    n_x=len(x)
    Mean_x=np.mean(x)
    SD_x=sqrt(np.var(x)) 
    #
    n_y=len(y)
    Mean_y=np.mean(y)
    SD_y=sqrt(np.var(y)) 
    corr=np.corrcoef(x,y)
    txt=r'$\rho_{x,y}}$=%.2f'+'\n $n$=%.0f \n $\mu_{x}$=%.1f \n $\sigma_{x}$=%.1f \n '
    txt+=' $\mu_{y}$=%.1f \n $\sigma_{y}$=%.1f'
    anchored_text = AnchoredText(txt %(corr[1,0], n_x,Mean_x,SD_x,Mean_y,SD_y), loc=loc,
                            prop={ 'size': font['size']*1.1, 'fontweight': 'bold'})
    ax1.add_artist(anchored_text)
    Lfunc1=polyfit(x,y,1)
    vEst=Lfunc1[0]*x+Lfunc1[1]
    try:
        title
    except NameError:
        pass  # do nothing! 
    else:
        plt.title(title,fontsize=font['size']*1.35)   
#
    try:
        xlabl
    except NameError:
        pass  # do nothing! 
    else:
        plt.xlabel(xlabl,fontsize=font['size']*scale)            
#
    try:
        ylabl
    except NameError:
        pass  # do nothing! 
    else:
        plt.ylabel(ylabl,fontsize=font['size']*scale)        
        
    try:
        xlimt
    except NameError:
        pass  # do nothing! 
    else:
        plt.xlim(xlimt)   
#        
    try:
        ylimt
    except NameError:
        pass  # do nothing! 
    else:
        plt.ylim(ylimt)   
      
    plt.plot(x,y,'ro',markersize=markersize,alpha=alpha,label='Not Missing')
    
    if(in_==1):
        plt.plot(x__1,x__2,'g*',markersize=markersize*1.6,label='Imputed',alpha=alpha)
    plt.legend(framealpha =10, loc=loc_1,markerscale=1.2) 
        


########################################################################################################### 

def plot_NN(history,ylim_1=[0.0, 1.0],ylim_2=[0.0, 1.0]):
    """ Plot training loss versus validation loss and 
    training accuracy versus validation accuracy"""
    
    font = {'size'   : 7.5}
    mpl.rc('font', **font)
    fig, ax=plt.subplots(figsize=(9, 3), dpi= 200, facecolor='w', edgecolor='k')
    
    ax1 = plt.subplot(1,2,1)    
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    ax1.plot(epochs, loss_values, 'bo', label='Training loss')          
    ax1.plot(epochs, val_loss_values, 'b', label='Validation loss')    
    plt.title('Training and validation loss',fontsize=11)
    plt.xlabel('Epochs (Early Stopping)',fontsize=9)
    plt.ylabel('Loss',fontsize=10)
    plt.legend(fontsize='8.5')
    plt.ylim(ylim_1)
    #
    ax2 = plt.subplot(1,2,2)    
    history_dict = history.history
    loss_values = history_dict['accuracy']
    val_loss_values = history_dict['val_accuracy']
    epochs = range(1, len(loss_values) + 1)
    ax2.plot(epochs, loss_values, 'ro', label='Training Accuracy')          
    ax2.plot(epochs, val_loss_values, 'r', label='Validation Accuracy')    
    plt.title('Training and validation Accuracy',fontsize=11)
    plt.xlabel('Epochs (Early Stopping)',fontsize=9)
    plt.ylabel('Accuracy',fontsize=10)
    plt.legend(fontsize='8.5')    
    plt.ylim(ylim_2)
    plt.show()
    
################################################################################################

def cond_LU_Sim(df,seed=45, normal_out=False,col_in=False):
    """
    Imputation of missing value by LU conditinal Simulation
    
    df: Pandas DataFrame with features with missing values (texts should be converted to numbers)
    seed: random number seed
    """
    
    def idx_non_missing_to_missing(values):
        """
        Move missing to the end of list
        """
        nan_in=[i1 for i1,j1 in enumerate (values) if np.isnan(j1)]
        idx=[i1 for i1,j1 in enumerate (values) if i1 not in nan_in] + nan_in
        return idx
    
    #######################################
    
    def L_Chol(df,idx):
        """
        Calculate Cholesky decomposition based on given index (idx)
        """
        clm=[df.columns[i1] for i1 in idx]
        corr=df[clm].corr().to_numpy()
        L=scipy.linalg.cholesky(corr, lower=True, overwrite_a=False)
        return L
    
    ########################################
    
    def nscore(df):
        """
        Nomal Score Transformation
        """
        colm_=list(df.columns)
        rows=len(df)
        ns_=np.zeros((rows,len(colm_)))
        for i in range(len(colm_)):
            ns=[]
            df_na=df[colm_[i]].to_numpy()
            df_nna=df[colm_[i]].dropna().to_numpy()
            val_=[]
            for j in range(len(df_na)):
                if np.isnan(df_na[j]):
                    val_.append(np.nan)
                else:    
                    val=percentileofscore(df_nna, df_na[j])
                    if val==100: val=val-0.001
                    if val==0:   val=val+0.001
                    val_.append(val/100)
            ns_[:,i]=norm.ppf(val_)  
        return ns_ 
    
    ########################################    
    
    # Number of rows and columns
    rows=len(df)
    colm_=list(df.columns)
    
    # Normal score transformation
    ns_=nscore(df) 
    
    # Assign random number seed
    np.random.seed(seed)
    
    # Change the correlation matrix to prevent having NaN for begainning
    # Cholesky decomposition is applied for each order of columns
    # to calculate lower matrix
    cols = df.columns.tolist()
    cols_L=np.arange(0,len(cols)).tolist()
    cols_=[]
    cols_n=[]
    L_=[]
    for i in range(len(cols)-1,0,-1):
        tmp_cols = cols[-i:] + cols[:-i]
        tmp_cols_n= cols_L[-i:] + cols_L[:-i]
        cols_n.append(tmp_cols_n)    
        cols_.append(tmp_cols)
        
        corr=df[tmp_cols].corr().to_numpy()
        L_tmp=scipy.linalg.cholesky(corr, lower=True, overwrite_a=False)
        L_.append(L_tmp)
        
    # Cholesky decomposition
    corr=df.corr().to_numpy()
    L=scipy.linalg.cholesky(corr, lower=True, overwrite_a=False) 
    
    # convert panda to numpy
    x_=df.to_numpy()
    w=np.zeros((len(cols),rows))
    w_=np.zeros((len(cols),rows))
    lu_ns=np.zeros((len(df),len(colm_)))
    
    # Simulate uncorrelated Gaussian distribution (mean=0, s.d=1)
    # for each variable
    mu=0
    sigma=1
    for i in range(len(cols)):
        Dist = np.random.normal(mu, sigma, rows)
        w[i,:]=Dist
        
    # Conditional LU simulation: conditioning available data
    # impute missing data while respecting the correlation between features.
    idx_list=[]
    L_idx_list=[]
    for i in range(rows):
        tmp_=[i1 for i1,j1 in enumerate (ns_[i,:]) if np.isnan(j1)]  
        
    ####### If there is no missing variable ########
        if len(tmp_)==0:
            w__=[]
            for j in range(len(colm_)):             
                ns_x=ns_[i,j]
                tmp=L[j,:]
                L_no0=tmp[:j]
                sum_=np.matmul(L_no0,w__)  
                w_new=(ns_x-sum_)/L[j,j] 
                w__.append(w_new)            
            tmp=(np.matmul(L,w__))   
            for k in range(len(tmp)):
                lu_ns[i,k]=tmp[k]          
    ####### If all variables are missing ########  
        elif(len(tmp_)==len(colm_)):
            w__=[]
            for j in range(len(colm_)):
                w__.append(w[j,i])                    
            tmp=(np.matmul(L,w__))   
            for k in range(len(tmp)):
                lu_ns[i,k]=tmp[k]          
       ####### If there are more than one missing ########     
        elif (len(tmp_)>0):
            cols_idx=idx_non_missing_to_missing(ns_[i,:])
            if (i==0): 
                idx_list.append(cols_idx) 
                L_tmp=L_Chol(df,cols_idx)
                L_idx_list.append(L_tmp)        
            try:
                idx=idx_list.index(cols_idx)
                L_idx=L_idx_list[idx]
            except ValueError:
                idx_list.append(cols_idx)
                L_idx=L_Chol(df,cols_idx)
                L_idx_list.append(L_idx)             
            
            w__=[]
            ino=0    
            for j in cols_idx:
                if (np.isnan(ns_[i,j])):
                    w__.append(w[j,i])
                else:               
                    ns_x=ns_[i,j]
                    tmp=L_idx[ino,:]
                    L_no0=tmp[:ino]
                    sum_=np.matmul(L_no0,w__)  
                    w_new=(ns_x-sum_)/L_idx[ino,ino] 
                    w__.append(w_new) 
                ino+=1
            tmp=(np.matmul(L_idx,w__)) 
            jno=0 
            for k in cols_idx:
                lu_ns[i,k]=tmp[jno] 
                jno+=1             
                        
    # Convert normal score data to pandas dataframe       
    pd_={}
    for i in range(len(colm_)):
        pd_[colm_[i]]=lu_ns[:,i]
     
    pd_ns=pd.DataFrame(pd_,columns=colm_) 
    
    # Replace Normal Gaussian quantile of each imputed value with the quantile
    # of distributio of mean for each feature.
    np.random.seed(seed+100)
    pd_b={}  
    for i in range(len(colm_)):
        tmp1=df[colm_[i]].to_numpy()
        tmp2=pd_ns[colm_[i]].to_numpy()
        btr=[]
        
        value_=df[colm_[i]].dropna().values
    
        for j in range(len(tmp1)):
            if (np.isnan(tmp1[j])):
                prob=norm.cdf(tmp2[j])
                quantle=np.quantile(value_, prob, axis=0, keepdims=True)[0] 
                if col_in:
                    if(colm_[i] in col_in):
                        btr.append(int(quantle))
                    else:
                        btr.append(quantle)    
                else:
                    btr.append(quantle)                    
            else:
                btr.append(tmp1[j])    
        pd_b[colm_[i]]=btr
        
    # Convert final imputed data to pandas dataframe    
    df_im=pd.DataFrame(pd_b,columns=colm_)
    if(normal_out):
        return pd_ns,df_im    
    else:
        return df_im    
        
    
#######################################################################   
   
def Conf_Matrix(y_train,y_train_pred,perfect=0,axt=None,plot=True,title=False,
                t_fontsize=8.5,t_y=1.2,x_fontsize=6.5,y_fontsize=6.5,trshld=0.5):
    '''Plot confusion matrix'''
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_score, recall_score
    from matplotlib.offsetbox import AnchoredText
    from sklearn.metrics import accuracy_score
    
    if (y_train_pred.shape[1]==2):
        y_train_pred=[0 if y_train_pred[i][0]>trshld else 1 for i in range(len(y_train_pred))]
    elif (y_train_pred.shape[1]==1):
        y_train_pred=[1 if y_train_pred[i][0]>trshld else 0 for i in range(len(y_train_pred))] 
    else:    
        y_train_pred=[1 if i>trshld else 0 for i in y_train_pred]       
    conf_mx=confusion_matrix(y_train,y_train_pred)
    acr=accuracy_score(y_train,y_train_pred)
    conf_mx =confusion_matrix(y_train,y_train_pred)
    prec=precision_score(y_train,y_train_pred) # == TP/(TP+FP) 
    reca=recall_score(y_train,y_train_pred) # == TP/(TP+FN) ) 
    TN=conf_mx[0][0] ; FP=conf_mx[0][1]
    spec= TN/(TN+FP)        
    if(plot):
        ax1 = axt or plt.axes()
        
        if (perfect==1): y_train_pred=y_train
        
        x=['Predicted \n Negative', 'Predicted \n Positive']; y=['Actual \n Negative', 'Actual \n Positive']
        ii=0 
        im =ax1.matshow(conf_mx, cmap='jet', interpolation='nearest') 
        for (i, j), z in np.ndenumerate(conf_mx): 
            if(ii==0): al='TN= '
            if(ii==1): al='FP= '
            if(ii==2): al='FN= '
            if(ii==3): al='TP= '          
            ax1.text(j, i, al+'{:0.0f}'.format(z), color='w', ha='center', va='center', fontweight='bold',fontsize=6.5)
            ii=ii+1
     
        txt='$ Accuracy\,\,\,$=%.2f\n$Sensitivity$=%.2f\n$Precision\,\,\,\,$=%.2f\n$Specificity$=%.2f'
        anchored_text = AnchoredText(txt %(acr,reca,prec,spec), loc=10, borderpad=0)
        ax1.add_artist(anchored_text)    
        
        ax1.set_xticks(np.arange(len(x)))
        ax1.set_xticklabels(x,fontsize=x_fontsize,y=0.97, rotation='horizontal')
        ax1.set_yticks(np.arange(len(y)))
        ax1.set_yticklabels(y,fontsize=y_fontsize,x=0.035, rotation='horizontal') 
        
        #cbaxes = ax1.add_axes([0.92, 0.36, 0.027, 0.25]) 
        cbar =plt.colorbar(im,shrink=0.3,
                           label='Low                              High',orientation='vertical')   
        cbar.set_ticks([])
        plt.title(title,fontsize=t_fontsize,y=t_y)
    return acr, prec, reca, spec    
   

#######################################################################   
   
def DNN (neurons=50,loss="binary_crossentropy",activation="relu",Nout=1,optimizer='adam',
             metrics=['accuracy'],activation_out='sigmoid',init_mode='uniform',BatchOpt=False,dropout_rate=False):
    """ Function to run Neural Network for different hyperparameters"""
    np.random.seed(42)
    tf.random.set_seed(42)
    
    if(activation=='Leaky_ReLU'): activation = keras.layers.LeakyReLU(alpha=0.2)
        
    # create model
    model = keras.models.Sequential()
    
    # Input & Hidden Layer 1
    model.add(keras.layers.Dense(neurons,input_dim=std_x_train.shape[1], activation=activation, kernel_initializer=init_mode))
        
    # Hidden Layer 2
    model.add(keras.layers.Dense(neurons,activation=activation)) 
    if(BatchOpt): model.add(keras.layers.BatchNormalization())
    if(dropout_rate):  model.add(keras.layers.Dropout(dropout_rate))
        
    # Hidden Layer 3    
    model.add(keras.layers.Dense(neurons,activation=activation)) 
    if(BatchOpt): model.add(keras.layers.BatchNormalization())
    if(dropout_rate):  model.add(keras.layers.Dropout(dropout_rate))
    
    # Hidden Layer 4    
    model.add(keras.layers.Dense(neurons,activation=activation)) 
    if(BatchOpt): model.add(keras.layers.BatchNormalization())
    if(dropout_rate):  model.add(keras.layers.Dropout(dropout_rate))     
    
    # Output Layer 
    model.add(keras.layers.Dense(Nout,activation=activation_out)) 
        
    # Compile model
    model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
    return model

#######################################################################   
   
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds,
    title="Precision and Sensitivity versus Thresholds", x=False,loc=3):
    plt.plot(thresholds, precisions[:-1], "r--", label="Precision", linewidth=3)
    plt.plot(thresholds, recalls[:-1], "g-", label="Sensitivity", linewidth=3)
    plt.legend(loc="center right", fontsize=16) 
    plt.xlabel("Thresholds", fontsize=12) 
    plt.ylabel("Precision/Sensitivity", fontsize=12)     
    plt.title(title, fontsize=12)   
    plt.grid(linewidth='0.25')   
    if x: plt.axvline(x=x,linestyle='--',color='k', linewidth=3,label='Probability '+str(np.round(x,2)))
    plt.legend(loc=loc, ncol=3,fontsize=10,markerscale=1.2, edgecolor="black",framealpha=0.9)
    plt.axis([0, 1, 0, 1])      

#######################################################################   

def boots_epoch(x, y, model, n_splits=40,min_delta=1e-5,patience=5,batch_size=32,test_size=0.2):
    """
    bootstraping on training set to find optimum number of epoches to avoid overfitting
    """
    import logging
    tf.get_logger().setLevel(logging.ERROR)  # Disable tensorflow warning
    
    # Apply bootstraping sampling
    
    boot = StratifiedShuffleSplit(n_splits=n_splits,test_size=0.2, random_state=42)
    
    ES_epoches=[]
    for train_idx, validation_idx in boot.split(x, y):
        x_trn = x[train_idx]
        y_trn = y[train_idx]
        x_val = x[validation_idx]
        y_val = y[validation_idx]  
    
        # Early stopping to avoid overfitting
        monitor= keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta,patience=patience, mode='auto')
        history=model_NN.fit(x_trn,y_trn,batch_size=batch_size,validation_data=
                  (x_val,y_val),callbacks=[monitor],verbose=0,epochs=1000)
        
        history_dict = history.history
        loss_values = history_dict['loss']
        epoch = len(loss_values)-1        
        ES_epoches.append(epoch)
    
    return int(np.mean(ES_epoches))   

#######################################################################   
   
def DNN_K_Fold_Cross(x,y,model_NN,cv=3,batch_size=32,verbose=0,epochs=10):
    """
    Implement K-Fold cross validation for Neural Network
    """
    # Cross-validate
    # Use for StratifiedKFold classification
    kf = StratifiedKFold(cv, shuffle=True, random_state=42) 
    oos_pred = []
    y_real=[]
    # Must specify y StratifiedKFold for
    for train, test in kf.split(x,y):   
        x_trn = x[train]
        y_trn = y[train]
        #
        x_tst = x[test]
        y_tst = y[test]    
    
        # Deep Neural Network
        model_NN.fit(x_trn,y_trn,batch_size=batch_size,verbose=verbose,epochs=epochs)
    
        pred = list(np.ravel(np.array(model_NN.predict(x_tst), dtype='float')))
        oos_pred.append(pred)
        y_real.append(y_tst)
        
    oos_pred=np.concatenate(oos_pred).ravel()
    y_real=np.concatenate(y_real).ravel()
    return oos_pred,y_real

#######################################################################   
   
def replace_encod(df,cat_encoded):
    encod=[]
    ir=0
    for i in df:
        if(pd.isnull(i)):
            encod.append(np.nan)
        else:
            encod.append(cat_encoded[ir][0])
            ir+=1
    return encod      

#######################################################################   
   
def CrossPlot (x,y,in_,x__1,x__2,title,xlabl,ylabl,loc,xlimt,
               ylimt,markersize,x__nna=None,y__nna=None,axt=None,scale=0.8,alpha=0.1,loc_1=2):
    '''Cross plot between two variables'''
    ax1 = axt or plt.axes()
    x=np.array(x)
    y=np.array(y)    
    n_x=len(x)
    Mean_x=np.mean(x)
    SD_x=sqrt(np.var(x)) 
    #
    n_y=len(y)
    Mean_y=np.mean(y)
    SD_y=sqrt(np.var(y)) 
    corr=np.corrcoef(x,y)
    txt=r'$\rho_{x,y}}$=%.2f'+'\n $n$=%.0f \n $\mu_{x}$=%.0f \n $\sigma_{x}$=%.0f \n '
    txt+=' $\mu_{y}$=%.0f \n $\sigma_{y}$=%.0f'
    anchored_text = AnchoredText(txt %(corr[1,0], n_x,Mean_x,SD_x,Mean_y,SD_y), loc=loc,
                            prop={ 'size': font['size']*1.1, 'fontweight': 'bold'})
    ax1.add_artist(anchored_text)
    Lfunc1=polyfit(x,y,1)
    vEst=Lfunc1[0]*x+Lfunc1[1]
    try:
        title
    except NameError:
        pass  # do nothing! 
    else:
        plt.title(title,fontsize=font['size']*1.35)   
#
    try:
        xlabl
    except NameError:
        pass  # do nothing! 
    else:
        plt.xlabel(xlabl,fontsize=font['size']*scale)            
#
    try:
        ylabl
    except NameError:
        pass  # do nothing! 
    else:
        plt.ylabel(ylabl,fontsize=font['size']*scale)        
        
    try:
        xlimt
    except NameError:
        pass  # do nothing! 
    else:
        plt.xlim(xlimt)   
#        
    try:
        ylimt
    except NameError:
        pass  # do nothing! 
    else:
        plt.ylim(ylimt)   
      
    if x__nna is not None:
        plt.plot(x__nna,y__nna,'ro',markersize=markersize,alpha=alpha,label='Not Missing')
    else:
        plt.plot(x,y,'ro',markersize=markersize,alpha=alpha,label='Not Missing')
    if(in_==1):
        plt.plot(x__1,x__2,'g*',markersize=markersize*1.6,label='Imputed',alpha=alpha)
    plt.legend(framealpha =10, loc=loc_1,markerscale=1.5) 
