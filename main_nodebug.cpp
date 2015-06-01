#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>


#include <fcntl.h>    /* For O_RDWR */
#include <unistd.h>   /* For open(), creat() */
#include <stdio.h>
#include <stdlib.h>
#include <Math.h>
#include <iostream>
#include <fstream>

#include <sys/time.h>
#include <time.h>


using namespace Eigen;
using namespace std;
using Eigen::ArrayXXd ;


#define ArgCount 3
#define PRECISION 10
#define MAX_ITER 200
#define TOL 0.00000001 

struct size{
	int n;
	int p;
} dimensions;

struct results{
	MatrixXd S;	//since this is return, I would not set it
	MatrixXd W;
	int iterations;
} result;

MatrixXd  readInputData(int row,int column);

MatrixXd getMean(MatrixXd X,int n);
MatrixXd normalize(MatrixXd X,MatrixXd means,int rows);
MatrixXd devide(MatrixXd u,MatrixXd d,int cols);
MatrixXd generateRandomMatrix(int n);
MatrixXd _ica_par(MatrixXd X1,MatrixXd w_init,int max_iter,double tol);
MatrixXd _sym_decorrelation(MatrixXd w_init);
MatrixXd arrayMultiplierRowWise(MatrixXd u,ArrayXXd temp,int n);
ArrayXXd multiplyColumnWise(MatrixXd g_wtx,MatrixXd W);
void testAccuracy(MatrixXd W);

MatrixXd cube(MatrixXd xin);
MatrixXd cubed(MatrixXd xin);

void WriteResultToFile(MatrixXd S);


void printMatrix(MatrixXd M){
	cout<<M.transpose()<<endl;
}

//function to measure time
typedef unsigned long long timestamp_t;

static timestamp_t get_timestamp (){
      struct timeval now;
      gettimeofday (&now, NULL);
      return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
      }




/*
FastICA class
*/
class FastICA{
	
	private:
	
	int n_components;
	int max_iter;
	double tol;
	
	//this called by fit_transform
	MatrixXd _fit(MatrixXd X){
		
		//call fastica function
		return fastica(X,n_components,max_iter,tol);
	}
	
	public:
	MatrixXd mixing_;
	MatrixXd W;
	
	FastICA(int numOfComponents){
		
		n_components = numOfComponents;
		max_iter = MAX_ITER;
		tol = TOL;
	}
	
	MatrixXd fit_transform(MatrixXd X){
		return _fit(X);
	}
	
	MatrixXd fastica(MatrixXd X,int n_components,int max_iter, double tol);
	
	
};


/*
The fastica function
inputs:
MatrixXd X - input data matrix
int n_components - number of channels
int max_iter - maximum iterations
double tol - tolerance when converging

output:
MatrixXd S - computed sources
*/

MatrixXd FastICA::fastica(MatrixXd X,int n_components,int max_iter, double tol){
	
	//n=rows,p=columns
	int n,p;
	
	//take dimensions from global structure
	n = dimensions.n;
	p = dimensions.p;
	
	MatrixXd means(n,1);	//mean of each row
	MatrixXd u(n,n);	//u of svd
	MatrixXd d(n,1);	//d of svd
	MatrixXd K(n,n);	
	MatrixXd X1(n,p);
	MatrixXd w_init(n,n);	//random matrix
	MatrixXd W(n,n);	//result
	MatrixXd unmixedSignal;	//unmixing X using W
	MatrixXd S;		//final result,computed sources
	
	//computing mean for every row
	means = getMean(X,n);
	//substracting mean from every element
	X = normalize(X,means,n);
	
	//Now calculate svd
	//Should try some other svd methods to optimize
	JacobiSVD<MatrixXd> svd(X, ComputeThinU|ComputeThinV);
	u =svd.matrixU();
	d = svd.singularValues();
	
	u = devide(u,d,n);
	K = u.transpose();
	
	X1 = (K*X)*sqrt(p);
	
	w_init = generateRandomMatrix(n);
	
	//calling the _ica_par paralleld ica algorithm function
	//it will return W
	W = _ica_par(X1,w_init,max_iter,tol);
	//now we have mixed matrix W
	//We should save this somewhere
	
	result.W = W;	//save in global structure
	
	//if whiten
	//do these things
	unmixedSignal = (W*K)*X;
	S = unmixedSignal.transpose();
	WriteResultToFile(S);
	
	return S;
}

/*
Writing S into file
*/
void WriteResultToFile(MatrixXd S){
	ofstream myfile;
	myfile.open("result.txt");
	myfile<<S;
	myfile.close();
}

MatrixXd _sym_decorrelation(MatrixXd w_init){
	
	MatrixXd wt;
	MatrixXd W;
	int n = dimensions.n;
	int i,j;
	
	MatrixXd s(1,n);	//eigenvalues
	MatrixXd u(n,n);	//eigenvectors
	MatrixXcd values;	//complex array returned by eigenvalues
	MatrixXcd vectors;	//complex array returned by eigenvectors
	
	wt = w_init.transpose();
	W	= w_init * wt;
	
	/*
	Since Eigenvalue compute give complex structure
	We should parse it into MatrixXd type
	*/
	EigenSolver<MatrixXd> eigenSolver(W,true);	//initializing eigen solver
	
	values = eigenSolver.eigenvalues();
	for(i=0;i<n;i++){
		s(0,n-i-1)= values(i,0).real();
	}
	
	vectors = eigenSolver.eigenvectors();
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			u(i,j) = vectors(i,n-j-1).real();
		}
	}
	
	return (arrayMultiplierRowWise(u,(1/sqrt(s.array())),n) * u.transpose())*w_init;
}


/*
Multiply each row of u by temp
*/
MatrixXd arrayMultiplierRowWise(MatrixXd u,ArrayXXd temp,int n){
	ArrayXXd uArray = u.array();
	int i;
	for(i=0;i<n;i++){
		uArray.row(i) *= temp;
	}
	return uArray.matrix();
}


/*
Parallel ICA algorithm
*/

MatrixXd _ica_par(MatrixXd X1,MatrixXd w_init,int max_iter,double tol){
	
	//we want symmetrical corellation of w_init
	MatrixXd W;
	double p_;	//number of samples
	int i;
	ArrayXXd gwtx_into_x1transpose_p;
	ArrayXXd gwtx_into_W;
	MatrixXd input_to_symmetric;	//gwtx_into_x1transpose_p - gwtx_into_W
	
	MatrixXd W1;
	double lim;	//limit to check with tolerance
	bool success = false;
	
	W = _sym_decorrelation(w_init);
	p_ = (double)dimensions.p;
	
	
	//ica main loop
	for(i=0;i<max_iter;i++){
		
		MatrixXd gwtx, g_wtx,dotprod;
		
		dotprod = W*X1;
		gwtx = cube(dotprod);	//cube of matrix
		g_wtx = cubed(dotprod);	//derivation of cube function

		gwtx_into_x1transpose_p = (gwtx * X1.transpose()).array()/p_;
		gwtx_into_W = multiplyColumnWise(g_wtx,W);
		input_to_symmetric = (gwtx_into_x1transpose_p - gwtx_into_W).matrix();	//gwtx_into_x1transpose_p -gwtx_into_W

		//call sym_decorrelation
		W1 = _sym_decorrelation(input_to_symmetric);
		
		lim =  ((((((W1*W.transpose()).diagonal()).array()).abs()) - 1).abs()).maxCoeff();	//max(abs(abs(diag(dot(W1, W.T))) - 1))
		W = W1;
		cout<<"lim: "<<lim<<"\n";
		if(lim<tol){
			success = true;
			break;
		}
		
	}
	result.iterations = i+1;	//save iterations
	if(!success){
		cout<<"!!!!! did not converged, increase the max_iter count!!!!!"<<endl;
	}
	
	return W;
}

ArrayXXd multiplyColumnWise(MatrixXd g_wtx,MatrixXd W){
	ArrayXXd W_in = W;
	ArrayXXd g_wtx_in =  g_wtx;
	int n = W_in.cols();
	int i;
	for(i=0;i<n;i++){
		W_in.col(i)*=g_wtx_in;
	}
	return W_in;
}

MatrixXd cube(MatrixXd xin){
	
	ArrayXXd x = xin.array();	//convert to Array
	x*=(x*x);
	return x.matrix();
}

MatrixXd cubed(MatrixXd xin){
	ArrayXXd x = xin.array();	//convert to Array
	x*=x;
	xin =(3*x).matrix();	//3*x^2
	return getMean(xin,dimensions.n);
}

//generate random matrix
//for convinience I put the matrix same as python solution

MatrixXd generateRandomMatrix(int n){
	return MatrixXd::Random(n,n);
}


MatrixXd devide(MatrixXd u,MatrixXd d,int cols){
	//each column of u should devide by each row of d
	int i;
	for(i=0;i<cols;i++){
		u.col(i) /= d(i,0);
	}
	return u;
}

MatrixXd normalize(MatrixXd X,MatrixXd means,int rows){
	
	//do element vise operation for every element
	//convert it to array and do the task
	int i;
	for(i=0;i<rows;i++){
		X.row(i) = X.row(i).array() - means(i,0);
	}
	return X;
	
}

MatrixXd getMean(MatrixXd X,int n){
	MatrixXd  means(n,1);
	int i;
	for(i=0;i<n;i++){
		means(i,0) = X.row(i).mean();
	}
	return means;
	
}



int main( int argc, char *argv[])
{
	//standard out precision
	cout.precision(PRECISION);
	
	int row,column;
	
	//Observation matrix
	MatrixXd X;
	//Result
	MatrixXd _S;
	
	if ( argc != ArgCount ){ 
		// We print argv[0] assuming it is the program name
		// Should provide row and column with the program name
		cout<<"usage: "<< argv[0] <<" <row> <column>\n";
		return 0;
	}
	
	//Take rows and columns from command line arguments
	//We take this much of data from input
	row=atoi(argv[1]);
    column=atoi(argv[2]);
	
	//set global dimension structure
	//So we don't have to compute size of array each time
	dimensions.n = row;
	dimensions.p = column;
	
	//read input from file into X
	X = readInputData(row,column);
	
	//starting time
	timestamp_t t0 = get_timestamp();
	
	//create object from FastICA class
	FastICA ica = FastICA(row);
	
	//call fit_transform method
	//_S is computed sources
	_S = ica.fit_transform(X);
	
	//measure finished time
    timestamp_t t1 = get_timestamp();
    cout<<"ica took : "<<(t1 - t0) / 1000000.0L<<endl;

	//cout<<"_S = \n"<<_S<<"\nfrom : "<<result.iterations<<" iterations\n";
	cout<<"W = \n"<<result.W<<"\n";
	
	//testAccuracy(result.W);
	
	return 0;	
}

void testAccuracy(MatrixXd W){
	//Defining A : mixing matrix
	MatrixXd A(3,3);
	A << 1, 1, 1,
       0.5, 2, 1.0,
	   1.5, 1.0, 2.0;
	cout<<"A.W\n"<<A*W<<endl;
	cout<<"W - A' \n"<<W.array() - (A.inverse()).array()<<endl;
	
}




/*
Function to read input data and put them into the matrix
arguments
int row - number of rows to scan
int column - number of columns to read

return value - filled initialX
*/
MatrixXd  readInputData(int row,int column){
	
	MatrixXd  initialX(row,column);
	FILE * fp = fopen("data.txt","r");
	int  i,j;
	double temp;
  
  for(i=0;i<row;i++){
	  for(j=0;j<column;j++){
		  fscanf(fp,"%lf",&temp);
		  initialX(i,j) = temp;
	  }
	  while(fgetc(fp)!='\n');
	  
  }

  return initialX;
  
}
	