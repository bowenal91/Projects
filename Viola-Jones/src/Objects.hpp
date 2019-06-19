class Image {
    public:
        double val[64][64];
        bool isface; //True if the image is a face. False if it is not.
};

class Feature { //Only features consisting of two rectangles. First rectangle is either at the top or the left

    //Rectangle vertices numbered numbered as:
    //                         1   2
    //                         3   4
    
    public: 
        int x1[2]; 
        int x2[2];
        int y1[2];
        int y2[2];

        double theta;
        double p;
};

class Learner {
    public:
        Feature *classifier; //An array of features (weak learners) used for this weighted strong learner
        double *alpha;       //Array of weights corresponding to each weak learner
        double Theta;        //Shift value on strong learner so that there are no false negatives
};
