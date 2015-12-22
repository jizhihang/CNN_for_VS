#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/gpu-ops.h"
#include "cnn/expr.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <iostream>
#include <fstream>

using namespace std;
using namespace cnn;
using namespace cnn::expr;

int main(int argc, char** argv) {
	cnn::Initialize(argc, argv);

/*	if (argc == 2) {
		ifstream in("");
		boost::archive::text_iarchive ia(in);
		ia >> m;
	}*/

	// parameters
	ifstream fin("C:\\Data\\SemevalData\\SemEval.train.filter2.emb200_oov_crosstrigram_k30");
	ifstream fin_mt("C:\\Data\\SemevalData\\SemEval.train.allmtscore");

	unsigned INPUT_SIZE_MT = 15;
	//unsigned INPUT_SIZE_MT = 0;
	unsigned INPUT_SIZE = 35 * 35;
	unsigned DATA_SIZE = 2669;
	unsigned OUTPUT_SIZE = 2;
	unsigned BATCH_SIZE = 64;
	
	const unsigned HIDDEN1_SIZE = 256; //512 is out-of-memory
	const unsigned HIDDEN2_SIZE = 64;
	const unsigned ITERATIONS = 1000000;
	fin >> INPUT_SIZE >> DATA_SIZE;

	Model m;
	//SimpleSGDTrainer sgd(&m);
	MomentumSGDTrainer sgd(&m);

	Parameters* P_W1 = m.add_parameters({ HIDDEN1_SIZE, INPUT_SIZE + INPUT_SIZE_MT });
	Parameters* P_b1 = m.add_parameters({ HIDDEN1_SIZE });
	Parameters* P_W2 = m.add_parameters({ HIDDEN2_SIZE, HIDDEN1_SIZE });
	Parameters* P_b2 = m.add_parameters({ HIDDEN2_SIZE });
	Parameters* P_V = m.add_parameters({ OUTPUT_SIZE, HIDDEN2_SIZE });
	Parameters* P_a = m.add_parameters({ OUTPUT_SIZE });

	vector<cnn::real> x_values;// (INPUT_SIZE * DATA_SIZE);
	x_values.clear();
	vector<cnn::real> y_values;// (OUTPUT_SIZE * DATA_SIZE);
	y_values.clear();
	for (int i = 0; i < DATA_SIZE; ++i)
	{
		int label;
		fin >> label;
		if (label == 0)
		{
			y_values.push_back(cnn::real(1));
			y_values.push_back(cnn::real(0));
		}
		else
		{
			y_values.push_back(cnn::real(0));
			y_values.push_back(cnn::real(1));
		}

		for (int j = 0; j < INPUT_SIZE_MT; ++j)
		{
			double x;
			fin_mt >> x;
			x_values.push_back(cnn::real(x));
		}
		for (int j = 0; j < INPUT_SIZE; ++j)
		{
			double x;
			fin >> x;
			x_values.push_back(cnn::real(x));
		}
	}
	fin.close();
	fin_mt.close();

	cerr << x_values.size() << '\n' << y_values.size() << '\n';
	//cerr << "x_dim=" << x_dim << ", y_dim=" << y_dim << endl;


	//Load dev data
	//ifstream f_test("C:\\Data\\msr_train.txt");
	ifstream f_test("C:\\Data\\SemevalData\\SemEval.dev.filter2.emb200_oov_crosstrigram_k30");
	ifstream f_test_mt("C:\\Data\\SemevalData\\SemEval.dev.allmtscore");

	vector<cnn::real> x_test_values;// (INPUT_SIZE * DATA_SIZE);
	x_test_values.clear();
	vector<cnn::real> y_test_values;// (OUTPUT_SIZE * DATA_SIZE);
	y_test_values.clear();
	unsigned TEST_SIZE;
	f_test >> INPUT_SIZE >> TEST_SIZE;
	for (int i = 0; i < TEST_SIZE; ++i)
	{
		int label;
		f_test >> label;
		if (label == 0)
		{
			y_test_values.push_back(cnn::real(1));
			y_test_values.push_back(cnn::real(0));
		}
		else
		{
			y_test_values.push_back(cnn::real(0));
			y_test_values.push_back(cnn::real(1));
		}
		for (int j = 0; j < INPUT_SIZE_MT; ++j)
		{
			double x;
			f_test_mt >> x;
			x_test_values.push_back(cnn::real(x));
		}
		for (int j = 0; j < INPUT_SIZE; ++j)
		{
			double x;
			f_test >> x;
			x_test_values.push_back(cnn::real(x));
		}
	}
	f_test_mt.close();
	f_test.close();


	double max = 0;
	int ki = 0;

	vector<unsigned> order((DATA_SIZE + BATCH_SIZE - 1) / BATCH_SIZE);
	unsigned si = order.size();
	for (unsigned i = 0; i < order.size(); ++i)
		order[i] = i * BATCH_SIZE;

	for (unsigned iter = 0; iter < ITERATIONS; ++iter)
	{
		bool flag = false;

		// train the parameters
		{

			if (si == order.size())
			{
				si = 0;
				if (iter != 0)
				{
					sgd.update_epoch();
				}
				cerr << "**SHUFFLE\n";
				shuffle(order.begin(), order.end(), *rndeng);
				flag = true;
			}

			//cerr << '\n' << si << '\n';
			unsigned r = order[si];
			unsigned batchsize = std::min((DATA_SIZE - r), BATCH_SIZE);
			vector<cnn::real> x((INPUT_SIZE + INPUT_SIZE_MT) * batchsize);
			vector<cnn::real> y(2 * batchsize);

			for (unsigned i = 0; i < batchsize; ++i)
			{
				//Notes: not suitable for this data set.
				//But seems better than random
				for (unsigned j = 0; j < INPUT_SIZE + INPUT_SIZE_MT; ++j)
				{
					x[i * (INPUT_SIZE + INPUT_SIZE_MT) + j] = x_values[(r + i) * (INPUT_SIZE + INPUT_SIZE_MT) + j];
				}
				y[i * 2] = y_values[(r + i) * 2];
				y[i * 2 + 1] = y_values[(r + i) * 2 + 1];
				
				/*
				r = rand() / DATA_SIZE;
				for (unsigned j = 0; j < INPUT_SIZE + INPUT_SIZE_MT; ++j)
				{
					x[i * (INPUT_SIZE + INPUT_SIZE_MT) + j] = x_values[r * (INPUT_SIZE + INPUT_SIZE_MT) + j];
				}
				y[i * 2] = y_values[r * 2];
				y[i * 2 + 1] = y_values[r * 2 + 1];
				*/
			}
			++si;

			Dim x_dim({ INPUT_SIZE + INPUT_SIZE_MT }, batchsize), y_dim({ OUTPUT_SIZE }, batchsize);

			ComputationGraph cg;

			Expression W1 = parameter(cg, P_W1);
			Expression b1 = parameter(cg, P_b1);
			Expression W2 = parameter(cg, P_W2);
			Expression b2 = parameter(cg, P_b2);
			Expression V = parameter(cg, P_V);
			Expression a = parameter(cg, P_a);

			// set x_values to change the inputs to the network
			Expression xr = input(cg, x_dim, &x);
			// set y_values expressing the output
			Expression yr = input(cg, y_dim, &y);

			Expression h1 = rectify(W1*xr + b1);
			Expression h2 = rectify(W2*h1 + b2);
			Expression y_pred = softmax(V*h2 + a);
			Expression loss = squared_distance(y_pred, yr);
			cnn::real labda = 1e-3;
			cnn::real re = m.gradient_l2_norm();
			cerr << re;
			Expression sum_loss = sum_batches(loss);// +labda * re;

			//cg.PrintGraphviz();

			float my_loss = as_scalar(cg.forward());
			cg.backward();
			if (iter < 10000)
				sgd.update(1e-1);
			else
				sgd.update(1e-4);
			//cerr << "ITERATIONS = " << iter << '\t';
			cerr << "E = " << my_loss << '\t'; //P = 1, iter = 6000, l_rate = 1
		}

		//DEV SCORE
		//if (flag)
		{
			flag = false;

			double l = 0;
			vector<double> pred_result;
			pred_result.clear();
			for (int i = 0; i < TEST_SIZE; ++i)
			{
				ComputationGraph cgr;
				Expression Wr1 = parameter(cgr, P_W1);
				Expression br1 = parameter(cgr, P_b1);
				Expression Wr2 = parameter(cgr, P_W2);
				Expression br2 = parameter(cgr, P_b2);
				Expression Vr = parameter(cgr, P_V);
				Expression ar = parameter(cgr, P_a);

				vector<cnn::real> x(INPUT_SIZE + INPUT_SIZE_MT);
				for (int j = 0; j < INPUT_SIZE + INPUT_SIZE_MT; ++j)
				{
					x[j] = x_test_values[j + i * (INPUT_SIZE + INPUT_SIZE_MT)];
				}
				Expression xr = input(cgr, { INPUT_SIZE + INPUT_SIZE_MT }, &x);
				vector<cnn::real> y(2);
				Expression yr = input(cgr, { OUTPUT_SIZE }, &y);
				Expression hr1 = rectify(Wr1*xr + br1);
				Expression hr2 = rectify(Wr2*hr1 + br2);
				Expression y_predr = softmax(Vr*hr2 + ar);
				y = as_vector(cgr.forward());

				double tmp;
				int t;
				if (y[0] > y[1])
				{
					tmp = y[0];
					t = 0;
				}
				else
				{
					tmp = y[1];
					t = 1;
				}
				int label = y_test_values[i * 2] == 1 ? 0 : 1;
				l += (t == label) ? 1 : 0;
				pred_result.push_back(tmp);
				pred_result.push_back(t);
			}
			l /= TEST_SIZE;
			cerr << "ITERATIONS = " << iter << '\t';
			cerr << "P = " << l << '\t';

			if (l > max)
			{
				max = l;
				ki = iter;
				ofstream fmodel("C:\\Data\\SemevalData\\model\\r=0.1_batchsize=64dropout_emb200_oov_crosstrigram_k30_filter2.model");
				boost::archive::text_oarchive oa(fmodel);
				oa << m;
				
				ofstream fout("C:\\Data\\SemevalData\\model\\modelr=0.1_batchsize=64dropour_emb200_oov_crosstrigram_k30_filter2.pred");
				for (unsigned i = 0; i < pred_result.size(); i += 2)
				{
					fout << pred_result[i] << '\t' << pred_result[i + 1] << '\n';
				}
				fout.close();
			}
			cerr << "max acc = " << max << "\tat\t" << ki << '\n';
		}
	}
	system("pause");
	return 0;
}

