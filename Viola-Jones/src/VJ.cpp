#include "time.h"
#include <iostream>
#include <math.h>
#include <cstdlib>
#include <stdio.h>
#include "VJ.hpp"
#include <algorithm>
#define PI 3.14159265358979323846

using namespace std;

VJ::VJ() {

}

void VJ::init_train() {
    specify_parameters(); //Parameters are specified
    initialize_objects(); 
    read_images();        //reads image data
    generate_features();  //Generates all features to be evaluated
    integrate_images();   //Integrates the images
}

void VJ::run() {
    int i;
    FILE *output = fopen("learner.txt","w");
    fclose(output);
    for (layer=0;layer<numLayers;layer++) {//Loop through all layers of the cascade
        printf("Building Cascade Layer %d\n",layer+1);
        create_images();//Generate the set of images to be used for this layer
        calc_params();  //Calculate theta and p for each feature on the current set of images
        adaboost();     //Select a strong classifier using AdaBoost
        get_fp();       //Find the false positives and calculate error rates
        write_learner(); //Print the data out to be used
    }


    //The following calculates the overall cascade error rate on the training data
    images.clear();
    fp.clear();
    for (i=0;i<numBackground;i++) {
        fp.push_back(i);
    }
    numImages = numFaces+numBackground;
    create_images();
    double value;
    double error = 0.0;
    double total = 0.0;
    bool face;
    for (i=0;i<numImages;i++) {
        face = false;
        for (layer=0;layer<numLayers;layer++) {
            
            value = classify_image(layer,i);
            if (value < 0.0) {
                break;
            }
            if (value > 0.0 && layer==numLayers-1) {
                face = true;
            }
        }
        if (face) { //Classified as face
            if (images[i].isface) {
                total += 1.0;    
            } else {
                error += 1.0;
                total += 1.0;
            }
        } else {
            if (images[i].isface) {
                error += 1.0;
                total += 1.0;
            } else {
                total += 1.0;
            }
        }
    
    }
    printf("Overall final error rate: %f\t%f\t%f\n",error,total,error/total);

}

void VJ::specify_parameters() {
    numFaces = 2000;            //Number of total faces in training set
    numBackground = 2000;       //Number of total backgrounds in training set
    stride_size = 4;            //How much feature sizes increment by. 64/stride_size should be an integer.
    numLayers = 5;              //Number of layers in the cascade
    numClassifiers = new int[numLayers];
    numClassifiers[0] = 15;       //Desired number of classifiers for each layer of the cascade
    numClassifiers[1] = 30;
    numClassifiers[2] = 45;
    numClassifiers[3] = 70;
    numClassifiers[4] = 100;
    calc_numFeatures();

}

void VJ::calc_numFeatures() {
    int width,height,i,j,rightShifts,downShifts;
    int pix = 64;
    int maxHeightMult = pix/stride_size;
    int maxWidthMult = pix/stride_size/2;
    numFeatures = 0;

    for (i=1;i<=maxHeightMult;i++) {
        for (j=1;j<=maxWidthMult;j++) {
            height = i*stride_size;
            width = j*stride_size*2; //Assuming two rectangles side by side
            rightShifts = (pix-width+1);
            downShifts = (pix-height+1);
            //if (height>24 && width>24) {
            //    continue;
            //}
            numFeatures += (rightShifts)*(downShifts);
            
        }
    }

    numFeatures *= 2; //Previously I only counted features with rectangle split by a vertical line. This accounts for the ones split by a horizontal line
    printf("Size of Feature Pool: %d\n",numFeatures);

}

void VJ::initialize_objects() { //Everything generated here is static.
    faces = new Image[numFaces];
    backgrounds = new Image[numBackground];
    features = new Feature[numFeatures];
    learners = new Learner[numLayers];

    int i;
    for (i=0;i<numBackground;i++) { //At zeroth layer all background images are a false positive because they have not been filtered yet
        fp.push_back(i);
    }
    numImages = numFaces+numBackground;

    for (i=0;i<numLayers;i++) {
        learners[i].classifier = new Feature[numClassifiers[i]]; //Initializes each layer of the learner to have the correct number of features
        learners[i].alpha = new double[numClassifiers[i]];
    }
}

void VJ::read_images() {
    FILE *open1,*open2;
    open1 = fopen("faces.dat","r");
    open2 = fopen("background.dat","r");
    int m,i,j;
    for (m=0;m<numFaces;m++) {
        for (i=0;i<64;i++) {
            for (j=0;j<64;j++) {
                fscanf(open1,"%lf",&faces[m].val[i][j]);
            }
        }
        faces[m].isface = true;
    }
    fclose(open1);
    for (m=0;m<numBackground;m++) {
        for (i=0;i<64;i++) {
            for (j=0;j<64;j++) {
                fscanf(open2,"%lf",&backgrounds[m].val[i][j]);
            }
        }
        backgrounds[m].isface = false;
    }
    fclose(open2);
} 

void VJ::generate_features() {
    int width,height,i,j,ii,jj,m,rightShifts,downShifts;
    int pix = 64;
    int maxHeightMult,maxWidthMult;
    int left,right,top,bottom,middle;
    

    m = 0;
    //Generate the features arranged horizontally
    maxHeightMult = pix/stride_size;
    maxWidthMult = pix/stride_size/2;

    for (ii=1;ii<=maxHeightMult;ii++) {
        for (jj=1;jj<=maxWidthMult;jj++) {
            height = ii*stride_size;
            width = jj*stride_size*2;
            //if (height>24 && width > 24) continue;
            rightShifts = (pix - width +1);
            downShifts = (pix - height+1);
            for (i=0;i<rightShifts;i++) {
                for (j=0;j<downShifts;j++) {
                    left = i-1;
                    right = left+width;
                    top = j-1;
                    bottom = top + height;
                    middle = (right-left)/2; 

                    features[m].x1[0] = left;
                    features[m].x1[1] = middle;
                    features[m].y1[0] = top;
                    features[m].y1[1] = bottom;

                    features[m].x2[0] = middle;
                    features[m].x2[1] = right;
                    features[m].y2[0] = top;
                    features[m].y2[1] = bottom;

                    m++;
                }
            }

        }
    }

    //Generate the features arranged vertically

    maxHeightMult = pix/stride_size/2;
    maxWidthMult = pix/stride_size;

    for (ii=1;ii<=maxHeightMult;ii++) {
        for (jj=1;jj<=maxWidthMult;jj++) {
            height = ii*stride_size*2;
            width = jj*stride_size;
            rightShifts = (pix - width +1);
            downShifts = (pix - height+1);
            //if (height > 24 && width > 24) continue;
            for (i=0;i<rightShifts;i++) {
                for (j=0;j<downShifts;j++) {
                    left = i-1;
                    right = left+width;
                    top = j-1;
                    bottom = top + height;
                    middle = (bottom-top)/2; 

                    features[m].x1[0] = left;
                    features[m].x1[1] = right;
                    features[m].y1[0] = middle;
                    features[m].y1[1] = bottom;

                    features[m].x2[0] = left;
                    features[m].x2[1] = right;
                    features[m].y2[0] = top;
                    features[m].y2[1] = middle;
                    
                    m++;
                }
            }

        }
    }

}

void VJ::integrate_images() {
    int i,j,im;
    double s,ii,v1;
    double mean,stdev;

    //Below is for normalizing the images, which led to bad results.
    /*
    for (im=0;im<numFaces;im++) {
        mean = 0.0;
        stdev = 0.0;
        for (i=0;i<64;i++) {
            for (j=0;j<64;j++) {
                mean += faces[im].val[i][j];
                stdev += faces[im].val[i][j]*faces[im].val[i][j];
            }
        }

        mean /= (64.0*64.0);
        stdev /= (64.0*64.0);
        stdev = stdev - mean*mean;
        stdev = sqrt(stdev);

        for (i=0;i<64;i++) {
            for (j=0;j<64;j++) {
                faces[im].val[i][j] = (faces[im].val[i][j]-mean)/stdev;
            }
        }
    }
    for (im=0;im<numBackground;im++) {
        mean = 0.0;
        stdev = 0.0;
        for (i=0;i<64;i++) {
            for (j=0;j<64;j++) {
                mean += backgrounds[im].val[i][j];
                stdev += backgrounds[im].val[i][j]*backgrounds[im].val[i][j];
            }
        }

        mean /= (64.0*64.0);
        stdev /= (64.0*64.0);
        stdev = stdev - mean*mean;
        stdev = sqrt(stdev);

        for (i=0;i<64;i++) {
            for (j=0;j<64;j++) {
                backgrounds[im].val[i][j] = (backgrounds[im].val[i][j]-mean)/stdev;
            }
        }
    }
*/   
    for (im=0;im<numFaces;im++) {
        
        for (i=0;i<64;i++) { //Loop through each row
            s = 0.0;
            for (j=0;j<64;j++) {
                s += faces[im].val[i][j]; //Cumulative row sum
                if (i==0) { //If we are in the first row, then we are adding on zero
                    v1 = 0;
                } else {
                    v1 = faces[im].val[i-1][j];
                }

                faces[im].val[i][j] = v1+s;
            }
        }

    }

    for (im=0;im<numBackground;im++) {
        
        for (i=0;i<64;i++) { //Loop through each row
            s = 0.0;
            for (j=0;j<64;j++) {
                s += backgrounds[im].val[i][j]; //Cumulative row sum
                if (i==0) { //If we are in the first row, then we are adding on zero
                    v1 = 0;
                } else {
                    v1 = backgrounds[im].val[i-1][j];
                }

                backgrounds[im].val[i][j] = v1+s;
            }
        }

    }

}

void VJ::create_images() { //Initializes the list of images as all the faces and backgrounds. This list is modified at the end of each adaboost
    int i,m;
    for (i=0;i<numFaces;i++) {  //We require no false negatives, so all the faces need to be in the current layer
        images.push_back(faces[i]);
    }
    

    m = 0;
    for (i=numFaces;i<numImages;i++) {
        images.push_back(backgrounds[fp[m]]);
        m++;
    }

}

void VJ::calc_params() {//This function calculates the best threshold and polarity choice for each feature on the current set of images
    printf("Calculating Parameters for Layer %d\n",layer+1);
    
    int i,j,k,m;
    int minIndex,minPlus,minMinus;
    double polarity,errorPlus,errorMinus,thresh;
    double currErrorPlus, currErrorMinus;
    
    double Splus,Sminus,Tplus,Tminus;
    for (m=0;m<numFeatures;m++) {
        //printf("%d\n",m);
        vector<double> fitness;
        vector<int> index;
            
        //Initially run through each image and calculate its fitness
        for (i=0;i<numImages;i++) {
            index.push_back(i);
            fitness.push_back(compute_feature(i,m));
        }

        vector<int> indexSort;
        vector<double> fitness2 = fitness;
        indexSort = sort_data(fitness);
        //indexSort = sort_data(index,fitness);

        Tplus = 0.0;
        Tminus = 0.0;
        Splus = 0.0;
        Sminus = 0.0;
        
        //Best threshold is between values that minimize error.

        for (i=0;i<numImages;i++) {
            if (images[indexSort[i]].isface) {
                Tplus += 1.0;
            } else {
                Tminus += 1.0;
            }
        }
        currErrorPlus = 1000000.0;
        currErrorMinus = 1000000.0;
        for (i=0;i<numImages;i++) {
            if (images[indexSort[i]].isface) {
                Splus += 1.0;
            } else {
                Sminus += 1.0;
            }

            errorPlus = Sminus + (Tplus-Splus); //Error associated with labeling all previous values as face
            errorMinus = Splus + (Tminus - Sminus); //Error associated with labeling all previous values as background
            if (errorPlus < currErrorPlus) {
                currErrorPlus = errorPlus;
                minPlus = i;
            }
            if (errorMinus < currErrorMinus) {
                currErrorMinus = errorMinus;
                minMinus = i;
            }



        }

        if (currErrorPlus < currErrorMinus) { //Everything below minPlus is labeled a face
            polarity = 1.0;
            if (minPlus == fitness2.size()-1) {
                thresh = fitness2[indexSort[minPlus]]+0.00001;
            } else {
                thresh = (fitness2[indexSort[minPlus]]+fitness2[indexSort[minPlus+1]])/2.0;
            }
        } else {                            //Everything above minMinus is labeled a face
            polarity = -1.0;
            if (minMinus == fitness2.size()-1) {
                thresh = fitness2[indexSort[minMinus]]+0.00001;
            } else{
                thresh = (fitness2[indexSort[minMinus]]+fitness2[indexSort[minMinus+1]])/2.0;
            }
        }
        
        features[m].p = polarity;
        features[m].theta = thresh;


         

    }

}

double VJ::compute_feature(int im, int fe) {
    double rect1,rect2;
    int x1,x2,y1,y2;
    double v1,v2,v3,v4;

    //First rectangle
    x1 = features[fe].x1[0];
    x2 = features[fe].x1[1];
    y1 = features[fe].y1[0];
    y2 = features[fe].y1[1];
    if (x1 == -1 && y1 == -1) {
        v1 = 0.0;
        v2 = 0.0;
        v3 = 0.0;
        v4 = images[im].val[y2][x2];
    } else if (x1 == -1) {
        v1 = 0.0;
        v3 = 0.0;
        v2 = images[im].val[y1][x2];
        v4 = images[im].val[y2][x2];
    } else if (y1 == -1) {
        v1 = 0.0;
        v2 = 0.0;
        v3 = images[im].val[y2][x1];
        v4 = images[im].val[y2][x2];
    } else {
        v1 = images[im].val[y1][x1];
        v2 = images[im].val[y2][x1];
        v3 = images[im].val[y1][x2];
        v4 = images[im].val[y2][x2];
    }
    rect1 = v4 - v3 - v2 + v1;
    
    //Second Rectangle
    x1 = features[fe].x2[0];
    x2 = features[fe].x2[1];
    y1 = features[fe].y2[0];
    y2 = features[fe].y2[1];
    if (x1 == -1 && y1 == -1) {
        v1 = 0.0;
        v2 = 0.0;
        v3 = 0.0;
        v4 = images[im].val[y2][x2];
    } else if (x1 == -1) {
        v1 = 0.0;
        v3 = 0.0;
        v2 = images[im].val[y1][x2];
        v4 = images[im].val[y2][x2];
    } else if (y1 == -1) {
        v1 = 0.0;
        v2 = 0.0;
        v3 = images[im].val[y2][x1];
        v4 = images[im].val[y2][x2];
    } else {
        v1 = images[im].val[y1][x1];
        v2 = images[im].val[y2][x1];
        v3 = images[im].val[y1][x2];
        v4 = images[im].val[y2][x2];
    }
    rect2 = v4 - v3 - v2 + v1;

    return rect2-rect1;

}

vector<int> VJ::sort_data(vector<double>& v) //Return a vector containing the indices of v whose values are arranged in ascending order
{
    vector<int> indices(v.size());
    iota(indices.begin(), indices.end(), 0u);
    sort(indices.begin(), indices.end(), [&](int lhs, int rhs) {
        return v[lhs] < v[rhs];
    });
    vector<int> res(v.size());
    for (int i = 0; i != indices.size(); ++i) {
        res[indices[i]] = i;
    }
    return res;
}

void VJ::adaboost() { //AdaBoost algorithm to select best features in order to get below desired false negative rate
    puts("Performing AdaBoost");
    
    int t,numFalse,i,m;
    int T = numClassifiers[layer];
    double w[T+1][numImages];
    double Alpha[T];
    numFalse = numImages - numFaces;
    double sum;
    double minError;
    int minID;
    double featureValue,e;
    double Theta;
    
    //Initialize the first set of weights
    for (i=0;i<numImages;i++) {
        w[0][i] = 0.5/double(numImages);
    }

    for (t=0;t<T;t++) {
        //Normalize the weights
        sum = 0.0; 
        for (i=0;i<numImages;i++) {
            sum += w[t][i];
        }
        for (i=0;i<numImages;i++) {
            w[t][i] /= sum;
        }

        //Loop through all the features (with already selected p and theta) to choose best weak classifier
        minError = 20.0;
        minID = 0;
        for (m=0;m<numFeatures;m++) {
            sum = 0.0;
            for (i=0;i<numImages;i++) {
                //Check to see if image is classified correctly
                featureValue = compute_feature(i,m);
                if (features[m].p*featureValue < features[m].p*features[m].theta) {//Feature classified as a face
                    if (!images[i].isface) { //Classified incorrectly -- error increases
                        sum += w[t][i];
                    }
                } else { //Feature classified as not a face
                    if (images[i].isface) { //Classified incorrectly -- error increases
                        sum += w[t][i];
                    }
                }
            }
            if (sum < minError) {
                minError = sum;
                minID = m;
            }

        }


        //Learner adds on feature with lowest error;
        learners[layer].classifier[t] = features[minID];
        Alpha[t] = 0.5*log((1.0-minError)/minError);
        for (i=0;i<numImages;i++) {
            featureValue = compute_feature(i,minID);
            if (features[minID].p*featureValue<features[minID].p*features[minID].theta) {//Classified as face
                if (images[i].isface) { //Classified correctly
                    w[t+1][i] = w[t][i]*exp(-Alpha[t]);
                } else {
                    w[t+1][i] = w[t][i]*exp(Alpha[t]);
                }
            } else {//Classified as background
                if (images[i].isface) { //Incorrect
                    w[t+1][i] = w[t][i]*exp(Alpha[t]);
                } else {
                    w[t+1][i] = w[t][i]*exp(-Alpha[t]);
                }
            }
        }

    }

    //Assign the proper weights to each weak learner in my strong classifier
    for (t=0;t<T;t++) {
        learners[layer].alpha[t] = Alpha[t];
    }
    learners[layer].Theta = 0.0;
    //Run through the images and shift strong learner so that there are no false negatives
    Theta = 0.0;
    minID=0;
    for (i=0;i<numImages;i++) {
         featureValue = classify_image(layer,i);
         if (featureValue < 0.0 && images[i].isface) { //False negative
            if (featureValue < Theta) { //This feature is more falsely negative than the previous
                Theta = featureValue;
                minID = i;
            }
         }
    }
    learners[layer].Theta = Theta;

}

double VJ::classify_image(int L, int im) { //Classifies image im using strong classifier of layer L
    int i,j,k;

    
    double rect1,rect2;
    int x1,x2,y1,y2;
    double v1,v2,v3,v4;
    double value;
    double sum = 0.0;
    for (i=0;i<numClassifiers[L];i++) {
        //First rectangle
        x1 = learners[L].classifier[i].x1[0];
        x2 = learners[L].classifier[i].x1[1];
        y1 = learners[L].classifier[i].y1[0];
        y2 = learners[L].classifier[i].y1[1];
        if (x1 == -1 && y1 == -1) {
            v1 = 0.0;
            v2 = 0.0;
            v3 = 0.0;
            v4 = images[im].val[y2][x2];
        } else if (x1 == -1) {
            v1 = 0.0;
            v3 = 0.0;
            v2 = images[im].val[y1][x2];
            v4 = images[im].val[y2][x2];
        } else if (y1 == -1) {
            v1 = 0.0;
            v2 = 0.0;
            v3 = images[im].val[y2][x1];
            v4 = images[im].val[y2][x2];
        } else {
            v1 = images[im].val[y1][x1];
            v2 = images[im].val[y2][x1];
            v3 = images[im].val[y1][x2];
            v4 = images[im].val[y2][x2];
        }
        rect1 = v4 - v3 - v2 + v1;
        
        //Second Rectangle
        x1 = learners[L].classifier[i].x2[0];
        x2 = learners[L].classifier[i].x2[1];
        y1 = learners[L].classifier[i].y2[0];
        y2 = learners[L].classifier[i].y2[1];
        if (x1 == -1 && y1 == -1) {
            v1 = 0.0;
            v2 = 0.0;
            v3 = 0.0;
            v4 = images[im].val[y2][x2];
        } else if (x1 == -1) {
            v1 = 0.0;
            v3 = 0.0;
            v2 = images[im].val[y1][x2];
            v4 = images[im].val[y2][x2];
        } else if (y1 == -1) {
            v1 = 0.0;
            v2 = 0.0;
            v3 = images[im].val[y2][x1];
            v4 = images[im].val[y2][x2];
        } else {
            v1 = images[im].val[y1][x1];
            v2 = images[im].val[y2][x1];
            v3 = images[im].val[y1][x2];
            v4 = images[im].val[y2][x2];
        }
        rect2 = v4 - v3 - v2 + v1;
        value = rect2-rect1;
       
        if (learners[L].classifier[i].p*value < learners[L].classifier[i].p*learners[L].classifier[i].theta) {//Classify as face
            value = 1.0;
        } else {
            value = -1.0;
        }

        sum += learners[L].alpha[i]*value;
        
    }
    sum -= learners[L].Theta;
    return sum;

}

void VJ::get_fp() { //Figure out the false positives from the current classifier and let them pass through to the next layer. Deletes the images array
    vector<int> dummyfp;
    int i;
    double fPositive = 0.0;
    double fNegative = 0.0;
    double error = 0.0;
    double total = 0.0;
    double value;
    for (i=0;i<fp.size();i++) {
        dummyfp.push_back(fp[i]);
    }
    fp.clear();
    for (i=0;i<numImages;i++) {
         value = classify_image(layer,i);
         if (value >= 0.0 && !images[i].isface) { //False positive -- These are okay
            fPositive += 1.0;
            fp.push_back(dummyfp[i-numFaces]);
            total+=1.0;
         } 
         if (value < 0.0 && images[i].isface) { //False negative -- These should not happen
            fNegative += 1.0;
            total+=1.0;
         }
         if (value >= 0.0 && images[i].isface) {//True postiive
            total += 1.0;
         }
         if (value < 0.0 && !images[i].isface) {
            total+=1.0;
         }
    }
    numImages = numFaces + int(fPositive);
    error = fPositive+fNegative;
    
    printf("Total Error Rate for Cascade Layer %d: %f\t%f\t%f\n",layer+1,total,error,error/total);
    printf("False Positive Rate: %f\t%f\n",fPositive,fPositive/total);
    printf("False Negative Rate: %f\t%f\n",fNegative,fNegative/total);
    images.clear();

}

void VJ::write_learner() {//Print out all the data for the current learner
    FILE *output = fopen("learner.txt","a");
    int i;
    fprintf(output,"%d\n",numClassifiers[layer]);
    for (i=0;i<numClassifiers[layer];i++) {
        fprintf(output,"%d\t%d\t%d\t%d\n",learners[layer].classifier[i].x1[0],learners[layer].classifier[i].x1[1],learners[layer].classifier[i].y1[0],learners[layer].classifier[i].y1[1]);
        fprintf(output,"%d\t%d\t%d\t%d\n",learners[layer].classifier[i].x2[0],learners[layer].classifier[i].x2[1],learners[layer].classifier[i].y2[0],learners[layer].classifier[i].y2[1]);
        fprintf(output,"%f\n",learners[layer].classifier[i].p);
        fprintf(output,"%f\n",learners[layer].classifier[i].theta);
        fprintf(output,"%f\n",learners[layer].alpha[i]);
    }

    fprintf(output,"%f\n",learners[layer].Theta);
    fclose(output);
}


/***************************** BELOW FUNCTIONS ARE FOR TESTING ******************************************/


void VJ::init_read() {
    FILE *data = fopen("image.dat","r");
    int i,j;
    fscanf(data,"%d",&numRows);
    fscanf(data,"%d",&numCols);
    main_image = new double *[numRows];
    for (i=0;i<numRows;i++) {
        main_image[i] = new double [numCols];
        for (j=0;j<numCols;j++) {
            fscanf(data,"%lf",&main_image[i][j]);
        }
    }


    fclose(data);
    images.push_back(window);
    numImages = 1;
    read_learner();
}

void VJ::read_learner() {
    FILE *output = fopen("learner.txt","r");
    int i;
    numLayers = 5;
    numClassifiers = new int[numLayers];
    learners = new Learner[numLayers];
    for (layer=0;layer<numLayers;layer++) {
        fscanf(output,"%d",&numClassifiers[layer]);
        learners[layer].classifier = new Feature[numClassifiers[layer]]; 
        learners[layer].alpha = new double[numClassifiers[layer]]; 
        for (i=0;i<numClassifiers[layer];i++) {
            fscanf(output,"%d%d%d%d",&learners[layer].classifier[i].x1[0],&learners[layer].classifier[i].x1[1],&learners[layer].classifier[i].y1[0],&learners[layer].classifier[i].y1[1]);
            fscanf(output,"%d%d%d%d",&learners[layer].classifier[i].x2[0],&learners[layer].classifier[i].x2[1],&learners[layer].classifier[i].y2[0],&learners[layer].classifier[i].y2[1]);
            fscanf(output,"%lf",&learners[layer].classifier[i].p);
            fscanf(output,"%lf",&learners[layer].classifier[i].theta);
            fscanf(output,"%lf",&learners[layer].alpha[i]);
        }

        fscanf(output,"%lf",&learners[layer].Theta);
    }
    fclose(output);
}

void VJ::detect() {
    int i,j;

    
    int rightShifts = numCols-64;
    int downShifts = numRows-64;
    double value;
    int numPos = 0;
    bool face;
    int left,top;
    for (top=0;top<downShifts;top++) {
        for (left=0;left<rightShifts;left++) { //Run through all positions of the window
            shift_window(left,top);
            face = true;
            for (layer=0;layer<numLayers;layer++) {//Run through the cascade and stop if detecting a nonface
                value = classify_image(layer,0);
                if (value < 0.0) {//Classified as a non-face
                    face = false;
                    break;
                }
            }
            if (face) {
                if(check_overlap(left,top)) {
                    x_store.push_back(left);
                    y_store.push_back(top);
                    numPos++;
                }
            }

        }
    }

    printf("%d\n",numPos);
    FILE *output = fopen("face_locations.dat","w");
    for (i=0;i<x_store.size();i++) {
        fprintf(output,"%d\t%d\n",y_store[i],x_store[i]);
    }
    fclose(output);

}

void VJ::shift_window(int left, int top) { //Shifts window of image analysis (left,top) denotes the top left corner of the current window
    int i,j;
    int a,b;
    double mean,stdev;
    for (i=0;i<64;i++) {
        for (j=0;j<64;j++) {
            a = left+i;
            b = top+j;
            window.val[j][i] = main_image[b][a];
        }
    }

    /*
    mean = 0.0;
    stdev = 0.0;
    for (i=0;i<64;i++) {
        for (j=0;j<64;j++) {
            mean += window.val[i][j];
            stdev += window.val[i][j]*window.val[i][j];
        }
    }

    mean /= (64.0*64.0);
    stdev /= (64.0*64.0);
    stdev = stdev - mean*mean;
    stdev = sqrt(stdev);
    for (i=0;i<64;i++) {
        for (j=0;j<64;j++) {
            window.val[i][j] = (window.val[i][j]-mean)/stdev;
        }
    }
*/
    double s,v1;
    //Integrates image to be used for feature evaluation
    for (i=0;i<64;i++) { //Loop through each row
        s = 0.0;
        for (j=0;j<64;j++) {
            s += window.val[i][j]; //Cumulative row sum
            if (i==0) { //If we are in the first row, then we are adding on zero
                v1 = 0;
            } else {
                v1 = window.val[i-1][j];
            }

            window.val[i][j] = v1+s;
        }
    }

    images.clear();
    images.push_back(window);
}

bool VJ::check_overlap(int left, int top) {//Returns true if there are no overlaps
    int i,j;
    int thresh =64;
    bool flag = true;
    for (i=0;i<x_store.size();i++) {
        if ( abs(left-x_store[i]) < thresh && abs(top-y_store[i]) < thresh) {
            flag = false;
            break;
        }
    }
    return flag;
}
