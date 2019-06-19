#include "Objects.hpp"
#include <vector>


class VJ {
	public:

        Image *faces;       //Permanent array containing all the faces
        Image *backgrounds; //Permanant array containing all the backgrounds
        std::vector<Image> images;      //Array that is changed for each layer of the cascade. Will contain all images (all faces and some bg) that passed through previous layer
        Feature *features;  //Array containing complete set of possible features
        Learner *learners;  //Array containing the set of strong learners to be used in the cascade 
        std::vector<int> fp;     //Array containing a list of background images that are false positives

        int numFaces;
        int numBackground;
        int numFeatures;
        int numLayers;
        int numImages;
        int *numClassifiers;
        int stride_size;
        int layer;          //Current Layer that we are working on
        
        std::vector<int> x_store;
        std::vector<int> y_store;
        Image window;
        int numRows,numCols;
        double **main_image;


        VJ();
        void init_train();
        void run();
        void specify_parameters();
        void initialize_objects();
        void read_images();
        void generate_features();
        void integrate_images();
        void calc_numFeatures();
        void create_images();
        void calc_params();
        double compute_feature(int,int);
        void adaboost();
        double classify_image(int,int);
        void get_fp();
        void write_learner();
        std::vector<int> sort_data(std::vector<double> &v);

        void init_read();
        void detect();
        void shift_window(int,int);
        void read_learner();
        bool check_overlap(int,int);
        
};
