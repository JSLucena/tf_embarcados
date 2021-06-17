#include <hellfire.h>
#include <noc.h>
#include <math.h>

#define N_PATTERNS		16
#define INPUT_NEURONS		7
#define HIDDEN_NEURONS		8
#define OUTPUT_NEURONS		4
#define ETA			0.30
#define ALPHA			0.90
#define EPSILON			0.05
#define INITIAL_WEIGHT_MAX	0.50
#define EPOCHS_MAX		100000
#define STATUS_CYCLES		1000

#define MASTER_CORE 0


struct neural_net_weights_s {
	float hidden_weights[(INPUT_NEURONS+1) * HIDDEN_NEURONS];
	float output_weights[(HIDDEN_NEURONS+1) * OUTPUT_NEURONS];
	float diff_hidden_weights[(INPUT_NEURONS+1) * HIDDEN_NEURONS];
	float diff_output_weights[(HIDDEN_NEURONS+1) * OUTPUT_NEURONS];
};

struct neural_net_s {
	char input[N_PATTERNS * INPUT_NEURONS];
	char target[N_PATTERNS * OUTPUT_NEURONS];
	float hidden[HIDDEN_NEURONS];
	float output[OUTPUT_NEURONS];
	struct neural_net_weights_s weights;
};

float sigmoid(float val)
{
	return 1.0 / (1.0 + exp(-val));
}

void load(struct neural_net_s *net)
{
	int i, j;
char input[N_PATTERNS * INPUT_NEURONS] = {
		1, 1, 1, 1, 1, 1, 0,	// 0
		0, 1, 1, 0, 0, 0, 0,	// 1
		1, 1, 0, 1, 1, 0, 1,	// 2
		1, 1, 1, 1, 0, 0, 1,	// 3
		0, 1, 1, 0, 0, 1, 1,	// 4
		1, 0, 1, 1, 0, 1, 1,	// 5
		1, 0, 1, 1, 1, 1, 1,	// 6
		1, 1, 1, 0, 0, 0, 0,	// 7
		1, 1, 1, 1, 1, 1, 1,	// 8
		1, 1, 1, 0, 0, 1, 1,	// 9
		1, 1, 1, 0, 1, 1, 1,	// A
		0, 0, 1, 1, 1, 1, 1,	// B
		1, 0, 0, 1, 1, 1, 0,	// C
		0, 1, 1, 1, 1, 0, 1,	// D
		1, 0, 0, 1, 1, 1, 1,	// E
		1, 0, 0, 0, 1, 1, 1	// F
	};

	char target[N_PATTERNS * OUTPUT_NEURONS] = {
		0, 0, 0, 0,
		0, 0, 0, 1,
		0, 0, 1, 0,
		0, 0, 1, 1,
		0, 1, 0, 0,
		0, 1, 0, 1,
		0, 1, 1, 0,
		0, 1, 1, 1,
		1, 0, 0, 0,
		1, 0, 0, 1,
		1, 0, 1, 0,
		1, 0, 1, 1,
		1, 1, 0, 0,
		1, 1, 0, 1,
		1, 1, 1, 0,
		1, 1, 1, 1
	};





	for (i = 0; i < N_PATTERNS; i++) {
		for (j = 0; j < INPUT_NEURONS; j++)
			net->input[i * INPUT_NEURONS + j] = input[i * INPUT_NEURONS + j];
		for (j = 0; j < OUTPUT_NEURONS; j++)
			net->target[i * OUTPUT_NEURONS + j] = target[i * OUTPUT_NEURONS + j];
	}
}

void infer(struct neural_net_s *net, int p)
{
	int i, j;
	float acc;

	// compute hidden layer activations
	for (i = 0; i < HIDDEN_NEURONS; i++) {
		acc = net->weights.hidden_weights[INPUT_NEURONS * HIDDEN_NEURONS + i];
		for (j = 0; j < INPUT_NEURONS; j++)
			acc += net->input[p * INPUT_NEURONS + j] * net->weights.hidden_weights[j * HIDDEN_NEURONS + i];
		net->hidden[i] = sigmoid(acc);
	}

	// compute output layer activations and calculate errors
	for (i = 0; i < OUTPUT_NEURONS; i++) {
		acc = net->weights.output_weights[HIDDEN_NEURONS * OUTPUT_NEURONS + i];
		for (j = 0; j < HIDDEN_NEURONS; j++)
			acc += net->hidden[j] * net->weights.output_weights[j * OUTPUT_NEURONS + i];
		net->output[i] = sigmoid(acc);
	}
}

void show_training(struct neural_net_s *net)
{
	int i, p;
	char buf[30];

	for (p = 0; p < N_PATTERNS; p++) {
		printf("\ninput (%02d): ", p);
		for (i = 0; i < INPUT_NEURONS; i++)
			printf("%d", net->input[p * INPUT_NEURONS + i]);
		printf(" target: ");
		for (i = 0; i < OUTPUT_NEURONS; i++)
			printf("%d", net->target[p * OUTPUT_NEURONS + i]);

		infer(net, p);

		printf(" output: ");
		for (i = 0; i < OUTPUT_NEURONS; i++) {
			ftoa(net->output[i], buf, 4);
			printf("%s ", buf);
		}
		printf("(");
		for (i = 0; i < OUTPUT_NEURONS; i++)
			(net->output[i] < 0.5) ? putchar('0') : putchar('1');
		printf(")");
	}
	printf("\n");
}
void init_weights(struct neural_net_s *net)
{
	int i,j;
	float rnd;
	// init hidden_weights
	for (i = 0; i < HIDDEN_NEURONS; i++) {
		for (j = 0; j <= INPUT_NEURONS; j++) {
			rnd = (random() % 100) / 100.0;
			net->weights.hidden_weights[j * HIDDEN_NEURONS + i] = 2.0 * (rnd - 0.5) * INITIAL_WEIGHT_MAX;
		}
	}

	// init output_weights
	for (i = 0; i < OUTPUT_NEURONS; i++) {
		for (j = 0; j <= HIDDEN_NEURONS; j++) {
			rnd = (random() % 100) / 100.0;
			net->weights.output_weights[j * OUTPUT_NEURONS + i] = 2.0 * (rnd - 0.5) * INITIAL_WEIGHT_MAX;
		}
	}
}
struct message_output_t
{
	float hidden_weights[(INPUT_NEURONS+1) * HIDDEN_NEURONS];
	float output_weights[(HIDDEN_NEURONS+1) * OUTPUT_NEURONS];
	float error;
};
void train(struct neural_net_s *net)
{
	int i, j, p, q, r;
	int rnd_index[N_PATTERNS];
	int epoch;
	float rnd, error, acc;
	float hidden_delta[HIDDEN_NEURONS];
	float output_delta[OUTPUT_NEURONS];
	float diff_hidden_weights[(INPUT_NEURONS+1) * HIDDEN_NEURONS];
	float diff_output_weights[(HIDDEN_NEURONS+1) * OUTPUT_NEURONS];
	float net_hidden_weights[(INPUT_NEURONS+1) * HIDDEN_NEURONS];
	float net_output_weights[(HIDDEN_NEURONS+1) * OUTPUT_NEURONS];
	char buf[30];

	for (p = 0; p < N_PATTERNS; p++)
		rnd_index[p] = p;

	// init hidden_weights and diff_hidden_weights
	for (i = 0; i < HIDDEN_NEURONS; i++) {
		for (j = 0; j <= INPUT_NEURONS; j++) {
			diff_hidden_weights[j * HIDDEN_NEURONS + i] = 0.0;
			rnd = (random() % 100) / 100.0;
			net->weights.hidden_weights[j * HIDDEN_NEURONS + i] = 2.0 * (rnd - 0.5) * INITIAL_WEIGHT_MAX;
		}
	}

	// init output_weights and diff_output_weights
	for (i = 0; i < OUTPUT_NEURONS; i++) {
		for (j = 0; j <= HIDDEN_NEURONS; j++) {
			diff_output_weights[j * OUTPUT_NEURONS + i] = 0.0;
			rnd = (random() % 100) / 100.0;
			net->weights.output_weights[j * OUTPUT_NEURONS + i] = 2.0 * (rnd - 0.5) * INITIAL_WEIGHT_MAX;
		}
	}
	printf("\ninitial outputs (not trained):");
	show_training(net);

	for (epoch = 1; epoch < EPOCHS_MAX; epoch++) {
		// randomize training pattern order
		for (p = 0; p < N_PATTERNS; p++) {
			q = random() % N_PATTERNS;
			r = rnd_index[p];
			rnd_index[p] = rnd_index[q];
			rnd_index[q] = r;
		}
		error = 0.0;

		// cycle through training patterns
		for (q = 0; q < N_PATTERNS; q++) {
			p = rnd_index[q];

			// feedforward: compute hidden layer activations
			for (i = 0; i < HIDDEN_NEURONS; i++) {
				acc = net->weights.hidden_weights[INPUT_NEURONS * HIDDEN_NEURONS + i];
				for (j = 0; j < INPUT_NEURONS; j++)
					acc += net->input[p * INPUT_NEURONS + j] * net->weights.hidden_weights[j * HIDDEN_NEURONS + i];
				net->hidden[i] = sigmoid(acc);
			}

			// feedforward: compute output layer activation errors
			for (i = 0; i < OUTPUT_NEURONS; i++) {
				acc = net->weights.output_weights[HIDDEN_NEURONS * OUTPUT_NEURONS + i];
				for (j = 0; j < HIDDEN_NEURONS; j++)
					acc += net->hidden[j] * net->weights.output_weights[j * OUTPUT_NEURONS + i];
				net->output[i] = sigmoid(acc);
				output_delta[i] = (net->target[p * OUTPUT_NEURONS + i] - net->output[i]) * net->output[i] * (1.0 - net->output[i]);
				error += 0.5 * (net->target[p * OUTPUT_NEURONS + i] - net->output[i]) * (net->target[p * OUTPUT_NEURONS + i] - net->output[i]);
			}

			// backpropagation of errors to hidden layer
			for (i = 0; i < HIDDEN_NEURONS; i++) {
				acc = 0.0;
				for (j = 0; j < OUTPUT_NEURONS; j++)
					acc += net->weights.output_weights[i * OUTPUT_NEURONS + j] * output_delta[j];
				hidden_delta[i] = acc * net->hidden[i] * (1.0 - net->hidden[i]);
			}

			// update weights (input to hidden)
			for (i = 0; i < HIDDEN_NEURONS; i++) {
				diff_hidden_weights[INPUT_NEURONS * HIDDEN_NEURONS + i] = ETA * hidden_delta[i] + ALPHA * diff_hidden_weights[INPUT_NEURONS * HIDDEN_NEURONS + i];
				net_hidden_weights[INPUT_NEURONS * HIDDEN_NEURONS + i] += diff_hidden_weights[INPUT_NEURONS * HIDDEN_NEURONS + i];
				for (j = 0; j < INPUT_NEURONS; j++) {
					diff_hidden_weights[j * HIDDEN_NEURONS + i] = ETA * net->input[p * INPUT_NEURONS + j] * hidden_delta[i] + ALPHA * diff_hidden_weights[j * HIDDEN_NEURONS + i];
					net_hidden_weights[j * HIDDEN_NEURONS + i] += diff_hidden_weights[j * HIDDEN_NEURONS + i] ;
				}
			}

			// update weights (hidden to output)
			for (i = 0; i < OUTPUT_NEURONS; i++) {
				diff_output_weights[HIDDEN_NEURONS * OUTPUT_NEURONS + i] = ETA * output_delta[i] + ALPHA * diff_output_weights[HIDDEN_NEURONS * OUTPUT_NEURONS + i];
				net_output_weights[HIDDEN_NEURONS * OUTPUT_NEURONS + i] += diff_output_weights[HIDDEN_NEURONS * OUTPUT_NEURONS + i];
				for (j = 0; j < HIDDEN_NEURONS; j++) {
					diff_output_weights[j * OUTPUT_NEURONS + i] = ETA * net->hidden[j] * output_delta[i] + ALPHA * diff_output_weights[j * OUTPUT_NEURONS + i];
					net_output_weights[j * OUTPUT_NEURONS + i] += diff_output_weights[j * OUTPUT_NEURONS + i];
				}
			}
		}
		
		// update network model weights (input to hidden)
		for (i = 0; i < HIDDEN_NEURONS; i++) {
			net->weights.hidden_weights[INPUT_NEURONS * HIDDEN_NEURONS + i] = net_hidden_weights[INPUT_NEURONS * HIDDEN_NEURONS + i];
			net->weights.diff_hidden_weights[INPUT_NEURONS * HIDDEN_NEURONS + i] = diff_hidden_weights[INPUT_NEURONS * HIDDEN_NEURONS + i];
			for (j = 0; j < INPUT_NEURONS; j++) {
				net->weights.hidden_weights[j * HIDDEN_NEURONS + i] = net_hidden_weights[j * HIDDEN_NEURONS + i] ;
				net->weights.diff_hidden_weights[j * HIDDEN_NEURONS + i] = diff_hidden_weights[j * HIDDEN_NEURONS + i] ;
			}
		}

		// update network model weights (hidden to output)
		for (i = 0; i < OUTPUT_NEURONS; i++) {
			net->weights.output_weights[HIDDEN_NEURONS * OUTPUT_NEURONS + i] = net_output_weights[HIDDEN_NEURONS * OUTPUT_NEURONS + i];
			net->weights.diff_output_weights[HIDDEN_NEURONS * OUTPUT_NEURONS + i] = diff_output_weights[HIDDEN_NEURONS * OUTPUT_NEURONS + i];
			for (j = 0; j < HIDDEN_NEURONS; j++) {
				net->weights.output_weights[j * OUTPUT_NEURONS + i] = net_output_weights[j * OUTPUT_NEURONS + i];
				net->weights.diff_output_weights[j * OUTPUT_NEURONS + i] = diff_output_weights[j * OUTPUT_NEURONS + i];
			}
		}

		if ((epoch % STATUS_CYCLES) == 0) {
			ftoa(error, buf, 6);
			printf("\nepoch: %d, error: %s", epoch, buf);
			show_training(net);
		}

		// if error rate is smaller than EPSILON, we are ok!
		if (error < EPSILON) break;
	}

	ftoa(error, buf, 6);
	printf("\n\ntraining set solved, %d epochs (error: %s)\n", epoch, buf);
}

void _main(void)
{
	char input[N_PATTERNS * INPUT_NEURONS] = {
		1, 1, 1, 1, 1, 1, 0,	// 0
		0, 1, 1, 0, 0, 0, 0,	// 1
		1, 1, 0, 1, 1, 0, 1,	// 2
		1, 1, 1, 1, 0, 0, 1,	// 3
		0, 1, 1, 0, 0, 1, 1,	// 4
		1, 0, 1, 1, 0, 1, 1,	// 5
		1, 0, 1, 1, 1, 1, 1,	// 6
		1, 1, 1, 0, 0, 0, 0,	// 7
		1, 1, 1, 1, 1, 1, 1,	// 8
		1, 1, 1, 0, 0, 1, 1,	// 9
		1, 1, 1, 0, 1, 1, 1,	// A
		0, 0, 1, 1, 1, 1, 1,	// B
		1, 0, 0, 1, 1, 1, 0,	// C
		0, 1, 1, 1, 1, 0, 1,	// D
		1, 0, 0, 1, 1, 1, 1,	// E
		1, 0, 0, 0, 1, 1, 1	// F
	};

	char target[N_PATTERNS * OUTPUT_NEURONS] = {
		0, 0, 0, 0,
		0, 0, 0, 1,
		0, 0, 1, 0,
		0, 0, 1, 1,
		0, 1, 0, 0,
		0, 1, 0, 1,
		0, 1, 1, 0,
		0, 1, 1, 1,
		1, 0, 0, 0,
		1, 0, 0, 1,
		1, 0, 1, 0,
		1, 0, 1, 1,
		1, 1, 0, 0,
		1, 1, 0, 1,
		1, 1, 1, 0,
		1, 1, 1, 1
	};

	struct neural_net_s neural_network;

	load(&neural_network);
	train(&neural_network);
	show_training(&neural_network);

	while(1);
}
void master(void)
{
	if (hf_comm_create(hf_selfid(), 1000 + MASTER_CORE, 0))
		panic(0xff);
	
	struct neural_net_s net;
	int i, p;
	char buf[300];
	uint16_t cpu, port, size;
	int32_t chan;
	int epoch;
	float error;
	
	
	struct message_output_t *accumulator;
	int proccount;
	
	//int receivedMessage[50];
	
	struct message_output_t * message_p;
	message_p = (struct message_output_t*)&buf[0];
	
	load(&net);
	init_weights(&net);
	
	printf("\ninitial outputs (not trained):");
	show_training(&net);
	
	// send initial weights to slaves
	for(i = 0; i < hf_ncores(); i++)
	{
		if(i != MASTER_CORE)
		{
			memcpy(buf, net.weights.hidden_weights, ((INPUT_NEURONS+1) * HIDDEN_NEURONS)*sizeof(float));
			hf_send(i, 1000 + i, buf, ((INPUT_NEURONS+1) * HIDDEN_NEURONS)*sizeof(float), MASTER_CORE);
			memcpy(buf, net.weights.output_weights, ((HIDDEN_NEURONS+1) * OUTPUT_NEURONS)*sizeof(float));
			hf_send(i, 1000 + i, buf, ((HIDDEN_NEURONS+1) * OUTPUT_NEURONS)*sizeof(float), MASTER_CORE);
		}
	}
	
	p = 0;
	epoch = 1;
	error = 99.9;
	
	while(1)
	{
		chan = hf_recvprobe();
		if(chan >= 0)
		{	
			hf_recv(&cpu, &port, buf, &size, chan);
			
			// send and recv hidden weights
			/*
			if(strcmp(buf, "hidden") == 0)
			{
				memcpy(buf, net.weights.hidden_weights, ((INPUT_NEURONS+1) * HIDDEN_NEURONS)*sizeof(float));
				hf_send(chan, 1000 + chan, buf, ((INPUT_NEURONS+1) * HIDDEN_NEURONS)*sizeof(float), MASTER_CORE);
				hf_recv(&cpu, &port, buf, &size, chan);
				memcpy(net.weights.hidden_weights, buf, ((INPUT_NEURONS+1) * HIDDEN_NEURONS)*sizeof(float));
				
			}
			* */

			for(i = 0; i < (INPUT_NEURONS+1) * HIDDEN_NEURONS; i++)
				accumulator->hidden_weights[i] += message_p->hidden_weights[i];
			for(i = 0; i < (HIDDEN_NEURONS+1) * OUTPUT_NEURONS; i++)
				accumulator->output_weights[i] += message_p->output_weights[i];
			accumulator->error += message_p->error;
			proccount++;
			
			
			
			if(proccount >= hf_ncores() -1)
			{
				proccount = 0;
				memcpy(buf, accumulator, (sizeof(struct message_output_t)));
				for(i = 1; i <= hf_ncores();i++)
				{
					printf("Sending to proc %d\n ", i);
					hf_send(i, 1000 + i, buf, ((INPUT_NEURONS+1) * HIDDEN_NEURONS)*sizeof(float), MASTER_CORE);
				}
					
			}		
			if ((epoch % STATUS_CYCLES) == 0) {
			ftoa(error, buf, 6);
			printf("\nepoch: %d, error: %s", epoch, buf);
			show_training(&net);
			}
				if (error < EPSILON) break;
			}
			
		}
		while(1);
}
	

void slave(void)
{
	if (hf_comm_create(hf_selfid(), 1000 + hf_cpuid(), 0))
		panic(0xff);
		
	struct neural_net_s *net;
	int i, j, p, q, r;
	int rnd_index[N_PATTERNS];
	char buf[300];
	
	
	float rnd, error, acc;
	float error_b;
	
	struct message_output_t * message_p;
	message_p = (struct message_output_t*)&buf[0];
	
	float diff_hidden_weights[(INPUT_NEURONS+1) * HIDDEN_NEURONS];
	float diff_output_weights[(HIDDEN_NEURONS+1) * OUTPUT_NEURONS];
	float net_hidden_weights[(INPUT_NEURONS+1) * HIDDEN_NEURONS];
	float net_output_weights[(HIDDEN_NEURONS+1) * OUTPUT_NEURONS];
	float hidden_delta[HIDDEN_NEURONS];
	float output_delta[OUTPUT_NEURONS];
	
	uint16_t cpu, port, size;
	
	load(&net);
	
	// init diff_hidden_weights
	for (i = 0; i < HIDDEN_NEURONS; i++) {
		for (j = 0; j <= INPUT_NEURONS; j++) {
			diff_hidden_weights[j * HIDDEN_NEURONS + i] = 0.0;
		}
	}

	// init diff_output_weights
	for (i = 0; i < OUTPUT_NEURONS; i++) {
		for (j = 0; j <= HIDDEN_NEURONS; j++) {
			diff_output_weights[j * OUTPUT_NEURONS + i] = 0.0;
		}
	}	
	
	// receive initial hidden weights
	hf_recv(&cpu, &port, buf, &size, MASTER_CORE);
	memcpy(net->weights.hidden_weights, buf, ((INPUT_NEURONS+1) * HIDDEN_NEURONS)*sizeof(float));
	// reveive initial outputs weights
	hf_recv(&cpu, &port, buf, &size, MASTER_CORE);
	memcpy(net->weights.output_weights, buf, ((HIDDEN_NEURONS+1) * OUTPUT_NEURONS)*sizeof(float));
	
	
	for (p = 0; p < N_PATTERNS; p++)
		rnd_index[p] = p;
	
	while(1)
	{
		for (p = 0; p < N_PATTERNS; p++) 
		{
			q = random() % N_PATTERNS;
			r = rnd_index[p];
			rnd_index[p] = rnd_index[q];
			rnd_index[q] = r;
		}
		error = 0.0;
		

		// cycle through training patterns
		for (q = 0; q < N_PATTERNS; q++) {
			p = rnd_index[q];

			// feedforward: compute hidden layer activations
			for (i = 0; i < HIDDEN_NEURONS; i++) {
				acc = net->weights.hidden_weights[INPUT_NEURONS * HIDDEN_NEURONS + i];
				for (j = 0; j < INPUT_NEURONS; j++)
					acc += net->input[p * INPUT_NEURONS + j] * net->weights.hidden_weights[j * HIDDEN_NEURONS + i];
				net->hidden[i] = sigmoid(acc);
			}

			// feedforward: compute output layer activation errors
			for (i = 0; i < OUTPUT_NEURONS; i++) {
				acc = net->weights.output_weights[HIDDEN_NEURONS * OUTPUT_NEURONS + i];
				for (j = 0; j < HIDDEN_NEURONS; j++)
					acc += net->hidden[j] * net->weights.output_weights[j * OUTPUT_NEURONS + i];
				net->output[i] = sigmoid(acc);
				output_delta[i] = (net->target[p * OUTPUT_NEURONS + i] - net->output[i]) * net->output[i] * (1.0 - net->output[i]);
				error += 0.5 * (net->target[p * OUTPUT_NEURONS + i] - net->output[i]) * (net->target[p * OUTPUT_NEURONS + i] - net->output[i]);
			}

			// backpropagation of errors to hidden layer
			for (i = 0; i < HIDDEN_NEURONS; i++) {
				acc = 0.0;
				for (j = 0; j < OUTPUT_NEURONS; j++)
					acc += net->weights.output_weights[i * OUTPUT_NEURONS + j] * output_delta[j];
				hidden_delta[i] = acc * net->hidden[i] * (1.0 - net->hidden[i]);
			}

			// update weights (input to hidden)
			for (i = 0; i < HIDDEN_NEURONS; i++) {
				diff_hidden_weights[INPUT_NEURONS * HIDDEN_NEURONS + i] = ETA * hidden_delta[i] + ALPHA * diff_hidden_weights[INPUT_NEURONS * HIDDEN_NEURONS + i];
				net_hidden_weights[INPUT_NEURONS * HIDDEN_NEURONS + i] += diff_hidden_weights[INPUT_NEURONS * HIDDEN_NEURONS + i];
				for (j = 0; j < INPUT_NEURONS; j++) {
					diff_hidden_weights[j * HIDDEN_NEURONS + i] = ETA * net->input[p * INPUT_NEURONS + j] * hidden_delta[i] + ALPHA * diff_hidden_weights[j * HIDDEN_NEURONS + i];
					net_hidden_weights[j * HIDDEN_NEURONS + i] += diff_hidden_weights[j * HIDDEN_NEURONS + i] ;
				}
			}

			// update weights (hidden to output)
			for (i = 0; i < OUTPUT_NEURONS; i++) {
				diff_output_weights[HIDDEN_NEURONS * OUTPUT_NEURONS + i] = ETA * output_delta[i] + ALPHA * diff_output_weights[HIDDEN_NEURONS * OUTPUT_NEURONS + i];
				net_output_weights[HIDDEN_NEURONS * OUTPUT_NEURONS + i] += diff_output_weights[HIDDEN_NEURONS * OUTPUT_NEURONS + i];
				for (j = 0; j < HIDDEN_NEURONS; j++) {
					diff_output_weights[j * OUTPUT_NEURONS + i] = ETA * net->hidden[j] * output_delta[i] + ALPHA * diff_output_weights[j * OUTPUT_NEURONS + i];
					net_output_weights[j * OUTPUT_NEURONS + i] += diff_output_weights[j * OUTPUT_NEURONS + i];
				}
			}
		}
		
		// update network model weights (input to hidden)
		for (i = 0; i < HIDDEN_NEURONS; i++) {
			net->weights.hidden_weights[INPUT_NEURONS * HIDDEN_NEURONS + i] = net_hidden_weights[INPUT_NEURONS * HIDDEN_NEURONS + i];
			net->weights.diff_hidden_weights[INPUT_NEURONS * HIDDEN_NEURONS + i] = diff_hidden_weights[INPUT_NEURONS * HIDDEN_NEURONS + i];
			for (j = 0; j < INPUT_NEURONS; j++) {
				net->weights.hidden_weights[j * HIDDEN_NEURONS + i] = net_hidden_weights[j * HIDDEN_NEURONS + i] ;
				net->weights.diff_hidden_weights[j * HIDDEN_NEURONS + i] = diff_hidden_weights[j * HIDDEN_NEURONS + i] ;
			}
		}

		// update network model weights (hidden to output)
		for (i = 0; i < OUTPUT_NEURONS; i++) {
			net->weights.output_weights[HIDDEN_NEURONS * OUTPUT_NEURONS + i] = net_output_weights[HIDDEN_NEURONS * OUTPUT_NEURONS + i];
			net->weights.diff_output_weights[HIDDEN_NEURONS * OUTPUT_NEURONS + i] = diff_output_weights[HIDDEN_NEURONS * OUTPUT_NEURONS + i];
			for (j = 0; j < HIDDEN_NEURONS; j++) {
				net->weights.output_weights[j * OUTPUT_NEURONS + i] = net_output_weights[j * OUTPUT_NEURONS + i];
				net->weights.diff_output_weights[j * OUTPUT_NEURONS + i] = diff_output_weights[j * OUTPUT_NEURONS + i];
			}
		}
		memcpy(message_p->hidden_weights, net->weights.hidden_weights, ((INPUT_NEURONS+1) * HIDDEN_NEURONS)*sizeof(float));
		memcpy(message_p->output_weights, net->weights.output_weights, ((HIDDEN_NEURONS+1) * OUTPUT_NEURONS)*sizeof(float));
		message_p->error=error;
		hf_send(MASTER_CORE, 1000+MASTER_CORE, buf, sizeof(struct message_output_t), hf_cpuid());
		
	}
}


void app_main(void)
{
	srand(hf_cpuid());
	
	int i;
	for(i = 0; i < hf_ncores(); i++)
	{
		if(hf_cpuid() == MASTER_CORE) {
			hf_spawn(master,0,0,0,"master",4096);
			break;
		}
		else {
			hf_spawn(slave,0,0,0,"slave",4096);
			break;
		}
	}
}
