#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>


//#include "src/redsvd.hpp"
//#include "src/redsvdFile.hpp"

#include <fcntl.h>    /* For O_RDWR */
#include <unistd.h>   /* For open(), creat() */
#include <stdio.h>
#include <stdlib.h>
#include <Math.h>
#include <iostream>
#include <fstream>

using namespace Eigen;
using namespace std;
using Eigen::ArrayXXd ;


#define ArgCount 3
#define PRECISION 10


ArrayXXd  readInputData(ArrayXXd  initialX,int row,int column);
ArrayXXd standarize(ArrayXXd initialX,int row,int column);
MatrixXd createXfromInput(int row,int column);
MatrixXd getMean(MatrixXd X,int n);
MatrixXd normalize(MatrixXd X,MatrixXd means,int rows);
MatrixXd devide(MatrixXd u,MatrixXd d,int cols);
MatrixXd generateRandomMatrix(int n);
MatrixXd _ica_par(MatrixXd X1,MatrixXd w_init,int max_iter,double tol);
MatrixXd _sym_decorrelation(MatrixXd w_init);
MatrixXd arrayMultiplierRowWise(MatrixXd u,ArrayXXd temp,int n);
MatrixXd getEigenVectors(MatrixXd W,int n);
MatrixXd getEigenValues(MatrixXd W,int n);
ArrayXXd multiplyColumnWise(MatrixXd g_wtx,MatrixXd W);

MatrixXd cube(MatrixXd xin);
MatrixXd cubed(MatrixXd xin);

//stat functions
double getMean(ArrayXXd data,int column);
double getVariance(ArrayXXd data,int column);
double getStdDev(ArrayXXd data,int column);
void testStatFunctions();

void printMatrix(MatrixXd M){
	cout<<M.transpose()<<endl;
}




/*
FastICA class
*/
class FastICA{
	
	public:
	MatrixXd mixing_;
	MatrixXd W;
	
	FastICA(int numOfComponents){
		
		n_components = numOfComponents;
		max_iter = 200;
		tol = 0.0001;
	}
	MatrixXd fit_transform(MatrixXd X){
		return _fit(X);
	}
	
	MatrixXd fastica(MatrixXd X,int n_components,int max_iter, double tol);
	
	private:
	
	int n_components;
	int max_iter;
	double tol;
	
	//this called by fit_transform
	MatrixXd _fit(MatrixXd X){
		
		//call fastica function
		return fastica(X,n_components,max_iter,tol);
	}
	
	
	
};


/*
The fastica function
*/

MatrixXd FastICA::fastica(MatrixXd X,int n_components,int max_iter, double tol){
	
	cout<<"You called fastica"<<endl;
	cout<<"I am printing X"<< X.rows() <<endl;
	printMatrix(X);
	//n = row,p = column
	//This is performance issu
	//best to pass the row column from begining
	int n,p;
	n = X.rows();
	p = X.cols();
	cout<<n<<" "<<p<<endl;
	
	//computing mean for every row
	MatrixXd means(n,1);
	means = getMean(X,n);
	
	cout<<"returned means are\n";
	printMatrix(means);
	
	//substracting mean from every element
	X = normalize(X,means,n);
	cout<<"normalized X is\n";
	printMatrix(X);
	
	//Now calculate svd
	JacobiSVD<MatrixXd> svd(X, ComputeThinU);
	cout<<"results of svd\n";
	
	MatrixXd u,d;
	
	u =svd.matrixU();
	d = svd.singularValues();
	
	cout<<d<<endl;
	cout<<u<<endl;
	/*
	cout<<"output of other redsvd\n";
	MatrixXf Y;
	REDSVD::RedSVD redsvd;
	redsvd.run(Y, 0);
	MatrixXf U = redsvd.matrixU();
	VectorXf S = redsvd.singularValues();
	MatrixXf V = redsvd.matrixV();
	cout<<"U = \n"<<U<<"\n";
	cout<<"S = \n"<<S<<"\n";
	cout<<"V = \n"<<V<<"\n";
	*/
	u = devide(u,d,n);
	cout <<"u after divide\n"<<u<<endl;
	
	MatrixXd K = u.transpose();
	
	//delete this part after development
	
	//K.row(0) = (-1*(K.row(0).array())).matrix();
	//K.row(1) = (-1*(K.row(1).array())).matrix();
	
	
	MatrixXd X1 = K*X;
	
	cout<<"X1 is\n";
	printMatrix(X1);
	//multiply by sqrt(p)
	
	X1 *= sqrt(p);
	cout<<"after multiplying by sqrt(p)"<<endl;
	printMatrix(X1);
	
	cout<<"initializing the random w_init"<<endl;
	
	MatrixXd w_init;
	//generate nxn matrix
	w_init = generateRandomMatrix(n);
	
	cout<<"We got the w_init"<<endl;
	cout<<w_init<<endl;
	
	//calling the _ica_par paralleld ica algorithm function
	//it will return W
	MatrixXd W;
	W = _ica_par(X1,w_init,max_iter,tol);
	
	
	//if whiten
	//do these things
	cout<<"W =\n"<<W<<"\n";
	cout<<"K =\n"<<K<<"\n";
	cout<<"X =\n"<<X<<"\n";
	MatrixXd innerDot = W*K;
	cout<<"fast_dot(W, K)\n"<<innerDot<<"\n";
	MatrixXd outerDot = innerDot*X;
	MatrixXd S = outerDot.transpose();
	cout<<"fast_dot(fast_dot(W, K), X)\n"<<S<<"\n";
	
	//writing output to file
	//to plot
	ofstream myfile;
	myfile.open("result.txt");
	myfile<<S;
	myfile.close();
	
	return S;
}

MatrixXd _sym_decorrelation(MatrixXd w_init){
	
	cout<<"finding sym decorreleation"<<endl;
	
	MatrixXd wt;
	MatrixXd W;
	//EigenvectorsType matV(w_init.cols(),w_init.cols());
	
	
	wt = w_init.transpose();
	W	= w_init * wt;
	cout<<"dot product is \n"<<W<<endl;
	//EigenSolver<MatrixXd> eigenSolver(W,true);
	//s = eigenSolver.eigenvalues();
	//u = eigenSolver.eigenvectors();
	int n = W.rows();
	MatrixXd s =getEigenValues(W,n);
	cout<<s<<endl;
	MatrixXd u = getEigenVectors(W,n);
	cout<<"u is\n"<<u<<endl;
	/*
	for debugging
	ArrayXXd temp;
	temp = (1/sqrt(s.array()));
	
	MatrixXd dotready = arrayMultiplierRowWise(u,temp,n); 
	cout<<"1/ sqrt of s\n"<<temp<<endl;
	cout<<"dotready \n"<< dotready<<endl;
	MatrixXd uT = u.transpose();
	MatrixXd temp2 = dotready * uT;
	MatrixXd temp3 = temp2*w_init;
	cout<<"dot product is \n"<< temp2<<"\n";
	cout<<"return value \n"<<temp3<<"\n";
	cout<<"or compute by one line \n results\n"<< (arrayMultiplierRowWise(u,(1/sqrt(s.array())),n) * u.transpose())*w_init<<"\n";
	*/
	cout<<"compute by one line \n results\n"<< (arrayMultiplierRowWise(u,(1/sqrt(s.array())),n) * u.transpose())*w_init<<"\n";
	return (arrayMultiplierRowWise(u,(1/sqrt(s.array())),n) * u.transpose())*w_init;
}

MatrixXd arrayMultiplierRowWise(MatrixXd u,ArrayXXd temp,int n){
	ArrayXXd uArray = u.array();
	int i;
	for(i=0;i<n;i++){
		uArray.row(i) *= temp;
	}
	return uArray.matrix();
}

/*
get eigenvalues

This should be optimized
*/

MatrixXd getEigenValues(MatrixXd W,int n){
	EigenSolver<MatrixXd> eigenSolver(W,true);
	MatrixXd s(1,n);
	MatrixXcd val = eigenSolver.eigenvalues();
	//cout<<"eigenvalues \n"<<val<<endl;
	int i;
	for(i=0;i<n;i++){
		s(0,n-i-1)= val(i,0).real();
	}
	return s;
}

MatrixXd getEigenVectors(MatrixXd W,int n){
	EigenSolver<MatrixXd> eigenSolver(W,true);
	MatrixXd u(n,n);
	MatrixXcd vectors=eigenSolver.eigenvectors();
	int i,j;
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			//u(i,j) = vectors.row(i).col(n-j-1).real();
			//cout<<vectors(i,n-j-1)<<" ";
			u(i,j) = vectors(i,n-j-1).real();
		}
		cout<<endl;
	}
	//cout<<"vectors\n"<<u<<endl;
	return u;

}


MatrixXd _ica_par(MatrixXd X1,MatrixXd w_init,int max_iter,double tol){
	
	//we want symmetrical corellation of w_init
	MatrixXd W;
	W = _sym_decorrelation(w_init);
	cout<<"We got w as \n"<<W<<"\n";
	double p_ = X1.cols();
	cout<<p_<<"\n";
	int i;
	
	
	
	
	//ica main loop
	for(i=0;i<max_iter;i++){
		
	
	MatrixXd gwtx, g_wtx,dotprod;
	cout<<"-----------------"<<"\n";
	
	cout<<"we are dot producting W and X1\n";
	cout<<"W = \n"<<W<<"\n";
	cout<<"X1 = \n"<<X1<<"\n";
	
	
	dotprod = W*X1;
	cout<<"Dotproduct = \n"<<dotprod<<"\n";
	cout<<"-------------------\n";
	cout<<dotprod<<"\n";
	gwtx = cube(dotprod);
	g_wtx = cubed(dotprod);
	cout<<"gwtx \n"<<gwtx<<"\n";
	cout<<"g_wtx \n"<<g_wtx<<"\n";
	
	
	/*
	 print "Calling sym_decorrelation function with (fast_dot(gwtx, X.T) / p_ - g_wtx[:, np.newaxis] * W)"
        print "fast_dot(gwtx, X.T)\n",fast_dot(gwtx, X.T)
        print "g_wtx[:, np.newaxis]\n",g_wtx[:, np.newaxis]
        print "g_wtx[:, np.newaxis] * W \n",g_wtx[:, np.newaxis] * W
        print "p_- g_wtx[:, np.newaxis] * W",p_ - g_wtx[:, np.newaxis] * W
       
	
	*/
	MatrixXd gwtx_into_x1transpose = gwtx * X1.transpose();
	ArrayXXd gwtx_into_x1transpose_p = gwtx_into_x1transpose.array()/p_;
	cout<<"gwtx * X1.transpose()\n"<<gwtx_into_x1transpose <<"\n";
	cout<<"gwtx * X1.transpose()/p_\n"<<gwtx_into_x1transpose_p <<"\n";
	cout<<"g_wtx[:, np.newaxis]\n"<<g_wtx <<"\n";
	
	ArrayXXd gwtx_into_W = multiplyColumnWise(g_wtx,W);
	cout<<"g_wtx[:, np.newaxis] * W\n"<<gwtx_into_W <<"\n";
	MatrixXd input_to_symmetric = (gwtx_into_x1transpose_p - gwtx_into_W).matrix();
	
	cout<<"fast_dot(gwtx, X.T) / p_- g_wtx[:, np.newaxis] * W\n"<<input_to_symmetric<<"\n";
	
	//call sym_decorrelation
	MatrixXd W1 = _sym_decorrelation(input_to_symmetric);
	cout<<"output of symdecorelation function \n"<<W1<<"\n";
	
	MatrixXd W1_dot_WT = W1*W.transpose();
	cout<<"W1_dot_WT\n"<<W1_dot_WT<<"\n";
	ArrayXXd W1_dot_W_diag = (W1_dot_WT.diagonal()).array();
	cout<<"W1_dot_W_diag\n"<<W1_dot_W_diag<<"\n";
	W1_dot_W_diag = W1_dot_W_diag.abs();
	cout<<"W1_dot_W_diag\n"<<W1_dot_W_diag<<"\n";
	W1_dot_W_diag = W1_dot_W_diag - 1;
	cout<<"W1_dot_W_diag -1\n"<<W1_dot_W_diag<<"\n";
	W1_dot_W_diag = W1_dot_W_diag.abs();
	cout<<"W1_dot_W_diag -1\n"<<W1_dot_W_diag<<"\n";
	
	W = W1;
	double lim =  W1_dot_W_diag.maxCoeff();
	cout<<"max value is \n"<<lim<<"\n";
	
	if(lim<tol){
		break;
	}
	//cout<<"devision \n"<<gwtx_into_x1transpose / p_gwtx_into_W <<"\n";
	
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
	cout<<"for cube my input \n"<<xin<<"\n";
	ArrayXXd x = xin.array();
	x*=(x*x);
	cout<<"first step \n"<<x<<"\n";
	cout<<"cube step \n"<<x<<"\n";
	return x.matrix();
	
}

MatrixXd cubed(MatrixXd xin){
	ArrayXXd x = xin.array();
	x*=x;
	xin =(3*x).matrix();
	return getMean(xin,xin.rows());
}

//generate random matrix
//for convinience I put the matrix same as python solution

MatrixXd generateRandomMatrix(int n){
	MatrixXd m(n,n);
	m << 1.76405235, 0.40015721, 0.97873798,
		2.2408932, 1.86755799, -0.97727788,
		0.95008842, -0.15135721,-0.10321885;
	return m;
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
	std::cout.precision(PRECISION);
	
	int row,column;
	
	//Observation matrix
	MatrixXd X;
	
	if ( argc != ArgCount ){ 
		// We print argv[0] assuming it is the program name
		// Should provide row and column with the program name
		std::cout<<"usage: "<< argv[0] <<" <row> <column>\n";
		return 0;
	}
	
	//take rows and columns from command line arguments
	row=atoi(argv[1]);
    column=atoi(argv[2]);

	X = createXfromInput(row,column);
	cout<<"your X matrix is\n";
	printMatrix(X);
	
	//create object from FastICA class
	FastICA ica = FastICA(row);
	//call fit_transform method
	MatrixXd _S;
	_S = ica.fit_transform(X);
	
	cout << "Your answer is \n";
	printMatrix(_S);
	return 0;	
}






/*
Crate X from input file and mixing matrix
*/

MatrixXd createXfromInput(int row,int column){
	
	//Defining A : mixing matrix
	MatrixXd A(3,3);
	A << 1, 1, 1,
       0.5, 2, 1.0,
	   1.5, 1.0, 2.0;
	std::cout << A<<std::endl ;
	
	ArrayXXd  initialX(row,column);
	initialX = readInputData(initialX,row,column);
	//print the matrix
	std::cout <<"Your S is \n"<<initialX.transpose() << std::endl;
	
	//testStatFunctions();
	
	ArrayXXd processedArray;
	//substract standard deviation
	processedArray	= standarize(initialX,row,column);
	
	std::cout<<"final array"<<endl<<processedArray.transpose()<<endl;
	
	//Mixing data with A
	//Converting processedArray into Matrix
	MatrixXd S;
	S = processedArray.matrix();
	
	MatrixXd _X;
	_X = A *S;
	
	return _X;
}


/*
function to divide std_Dev from each element
*/

ArrayXXd standarize(ArrayXXd initialX,int row,int column){
	
	//accessing row and calculate std_dev
	ArrayXXd  std_dev(row,1);
	int i;
	for(i=0;i<row;i++){
		//std::cout << "row "<< i <<initialX.row(i) << std::endl;
		std_dev(i,0) = getStdDev(initialX.row(i),column);
		//std::cout<<"std is "<< std_dev(i,0)<<std::endl;
	}
	for(i=0;i<column;i++){
		initialX.col(i) = initialX.col(i) / std_dev;
		
	}
	return initialX;
}

/*
Function to read input data and put them into the matrix
arguments
ArrayXXd  initialX - initialized input matrix [i][j]
int row - number of rows to scan_is
int column - number of columns to read

return value - filled initialX
*/
ArrayXXd  readInputData(ArrayXXd  initialX,int row,int column){
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




double getMean(ArrayXXd data,int column)
    {
        double sum = 0.0;
		int i=0;
        for(i=0;i<column;i++){
			sum += data(0,i);
		}
        return sum/column;
    }

    double getVariance(ArrayXXd data,int column)
    {
        double mean = getMean(data,column);
        double temp = 0;
		int a=0;
        for(a=0;a<column;a++)
            temp += (mean-data(0,a))*(mean-data(0,a));
        return temp/column;
    }

    double getStdDev(ArrayXXd data,int column)
    {
        return sqrt(getVariance(data,column));
    }
	
	
void testStatFunctions(){
	//testing the stat functions
	ArrayXXd test(1,10);
	test << 0.2292345 ,1.60985263,2.70559672,2.43609396,1.22700047,0.60039577,1.84131188,1.40126957,15.05481776,7.03233614;
		
		
	double out1 = getMean(test,10);
	std::cout<<"mean is "<<out1<<std::endl;
	
	double out2 = getVariance(test,10);
	std::cout<<"Variance is "<<out2<<std::endl;
	
	double out3 = getStdDev(test,10);
	cout<<"std is "<<out3<<endl;
	
}