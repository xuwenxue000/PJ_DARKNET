#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <ctype.h>
#include <strings.h>
#include <string.h>
#include <sys/stat.h>
#include <pthread.h>
#include <sys/wait.h>


#include "parser.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "connected_layer.h"

#include "option_list.h"
#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "blas.h"

#include"setjmp.h"
jmp_buf Jump_Buffer;
#define try if(!setjmp(Jump_Buffer))
#define catch else
#define throw longjmp(Jump_Buffer,1)


extern void predict_classifier(char *datacfg, char *cfgfile, char *weightfile, char *filename, int top);
extern void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen);
extern void run_voxel(int argc, char **argv);
extern void run_yolo(int argc, char **argv);
extern void run_detector(int argc, char **argv);
extern void run_coco(int argc, char **argv);
extern void run_writing(int argc, char **argv);
extern void run_captcha(int argc, char **argv);
extern void run_nightmare(int argc, char **argv);
extern void run_dice(int argc, char **argv);
extern void run_compare(int argc, char **argv);
extern void run_classifier(int argc, char **argv);
extern void run_regressor(int argc, char **argv);
extern void run_char_rnn(int argc, char **argv);
extern void run_vid_rnn(int argc, char **argv);
extern void run_tag(int argc, char **argv);
extern void run_cifar(int argc, char **argv);
extern void run_go(int argc, char **argv);
extern void run_art(int argc, char **argv);
extern void run_super(int argc, char **argv);
extern void run_lsd(int argc, char **argv);


void average(int argc, char *argv[])
{
    char *cfgfile = argv[2];
    char *outfile = argv[3];
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    network sum = parse_network_cfg(cfgfile);

    char *weightfile = argv[4];
    load_weights(&sum, weightfile);

    int i, j;
    int n = argc - 5;
    for(i = 0; i < n; ++i){
        weightfile = argv[i+5];
        load_weights(&net, weightfile);
        for(j = 0; j < net.n; ++j){
            layer l = net.layers[j];
            layer out = sum.layers[j];
            if(l.type == CONVOLUTIONAL){
                int num = l.n*l.c*l.size*l.size;
                axpy_cpu(l.n, 1, l.biases, 1, out.biases, 1);
                axpy_cpu(num, 1, l.weights, 1, out.weights, 1);
                if(l.batch_normalize){
                    axpy_cpu(l.n, 1, l.scales, 1, out.scales, 1);
                    axpy_cpu(l.n, 1, l.rolling_mean, 1, out.rolling_mean, 1);
                    axpy_cpu(l.n, 1, l.rolling_variance, 1, out.rolling_variance, 1);
                }
            }
            if(l.type == CONNECTED){
                axpy_cpu(l.outputs, 1, l.biases, 1, out.biases, 1);
                axpy_cpu(l.outputs*l.inputs, 1, l.weights, 1, out.weights, 1);
            }
        }
    }
    n = n+1;
    for(j = 0; j < net.n; ++j){
        layer l = sum.layers[j];
        if(l.type == CONVOLUTIONAL){
            int num = l.n*l.c*l.size*l.size;
            scal_cpu(l.n, 1./n, l.biases, 1);
            scal_cpu(num, 1./n, l.weights, 1);
                if(l.batch_normalize){
                    scal_cpu(l.n, 1./n, l.scales, 1);
                    scal_cpu(l.n, 1./n, l.rolling_mean, 1);
                    scal_cpu(l.n, 1./n, l.rolling_variance, 1);
                }
        }
        if(l.type == CONNECTED){
            scal_cpu(l.outputs, 1./n, l.biases, 1);
            scal_cpu(l.outputs*l.inputs, 1./n, l.weights, 1);
        }
    }
    save_weights(sum, outfile);
}

void speed(char *cfgfile, int tics)
{
    if (tics == 0) tics = 1000;
    network net = parse_network_cfg(cfgfile);
    set_batch_network(&net, 1);
    int i;
    time_t start = time(0);
    image im = make_image(net.w, net.h, net.c*net.batch);
    for(i = 0; i < tics; ++i){
        network_predict(net, im.data);
    }
    double t = difftime(time(0), start);
    printf("\n%d evals, %f Seconds\n", tics, t);
    printf("Speed: %f sec/eval\n", t/tics);
    printf("Speed: %f Hz\n", tics/t);
}

void operations(char *cfgfile)
{
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    int i;
    long ops = 0;
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.type == CONVOLUTIONAL){
            ops += 2l * l.n * l.size*l.size*l.c * l.out_h*l.out_w;
        } else if(l.type == CONNECTED){
            ops += 2l * l.inputs * l.outputs;
        }
    }
    printf("Floating Point Operations: %ld\n", ops);
    printf("Floating Point Operations: %.2f Bn\n", (float)ops/1000000000.);
}

void oneoff(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    int oldn = net.layers[net.n - 2].n;
    int c = net.layers[net.n - 2].c;
    scal_cpu(oldn*c, .1, net.layers[net.n - 2].weights, 1);
    scal_cpu(oldn, 0, net.layers[net.n - 2].biases, 1);
    net.layers[net.n - 2].n = 9418;
    net.layers[net.n - 2].biases += 5;
    net.layers[net.n - 2].weights += 5*c;
    if(weightfile){
        load_weights(&net, weightfile);
    }
    net.layers[net.n - 2].biases -= 5;
    net.layers[net.n - 2].weights -= 5*c;
    net.layers[net.n - 2].n = oldn;
    printf("%d\n", oldn);
    layer l = net.layers[net.n - 2];
    copy_cpu(l.n/3, l.biases, 1, l.biases +   l.n/3, 1);
    copy_cpu(l.n/3, l.biases, 1, l.biases + 2*l.n/3, 1);
    copy_cpu(l.n/3*l.c, l.weights, 1, l.weights +   l.n/3*l.c, 1);
    copy_cpu(l.n/3*l.c, l.weights, 1, l.weights + 2*l.n/3*l.c, 1);
    *net.seen = 0;
    save_weights(net, outfile);
}

void oneoff2(char *cfgfile, char *weightfile, char *outfile, int l)
{
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights_upto(&net, weightfile, 0, net.n);
        load_weights_upto(&net, weightfile, l, net.n);
    }
    *net.seen = 0;
    save_weights_upto(net, outfile, net.n);
}

void partial(char *cfgfile, char *weightfile, char *outfile, int max)
{
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights_upto(&net, weightfile, 0, max);
    }
    *net.seen = 0;
    save_weights_upto(net, outfile, max);
}

#include "convolutional_layer.h"
void rescale_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    int i;
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.type == CONVOLUTIONAL){
            rescale_weights(l, 2, -.5);
            break;
        }
    }
    save_weights(net, outfile);
}

void rgbgr_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    int i;
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.type == CONVOLUTIONAL){
            rgbgr_weights(l);
            break;
        }
    }
    save_weights(net, outfile);
}

void reset_normalize_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    int i;
    for (i = 0; i < net.n; ++i) {
        layer l = net.layers[i];
        if (l.type == CONVOLUTIONAL && l.batch_normalize) {
            denormalize_convolutional_layer(l);
        }
        if (l.type == CONNECTED && l.batch_normalize) {
            denormalize_connected_layer(l);
        }
        if (l.type == GRU && l.batch_normalize) {
            denormalize_connected_layer(*l.input_z_layer);
            denormalize_connected_layer(*l.input_r_layer);
            denormalize_connected_layer(*l.input_h_layer);
            denormalize_connected_layer(*l.state_z_layer);
            denormalize_connected_layer(*l.state_r_layer);
            denormalize_connected_layer(*l.state_h_layer);
        }
    }
    save_weights(net, outfile);
}

layer normalize_layer(layer l, int n)
{
    int j;
    l.batch_normalize=1;
    l.scales = calloc(n, sizeof(float));
    for(j = 0; j < n; ++j){
        l.scales[j] = 1;
    }
    l.rolling_mean = calloc(n, sizeof(float));
    l.rolling_variance = calloc(n, sizeof(float));
    return l;
}

void normalize_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    int i;
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.type == CONVOLUTIONAL && !l.batch_normalize){
            net.layers[i] = normalize_layer(l, l.n);
        }
        if (l.type == CONNECTED && !l.batch_normalize) {
            net.layers[i] = normalize_layer(l, l.outputs);
        }
        if (l.type == GRU && l.batch_normalize) {
            *l.input_z_layer = normalize_layer(*l.input_z_layer, l.input_z_layer->outputs);
            *l.input_r_layer = normalize_layer(*l.input_r_layer, l.input_r_layer->outputs);
            *l.input_h_layer = normalize_layer(*l.input_h_layer, l.input_h_layer->outputs);
            *l.state_z_layer = normalize_layer(*l.state_z_layer, l.state_z_layer->outputs);
            *l.state_r_layer = normalize_layer(*l.state_r_layer, l.state_r_layer->outputs);
            *l.state_h_layer = normalize_layer(*l.state_h_layer, l.state_h_layer->outputs);
            net.layers[i].batch_normalize=1;
        }
    }
    save_weights(net, outfile);
}

void statistics_net(char *cfgfile, char *weightfile)
{
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    int i;
    for (i = 0; i < net.n; ++i) {
        layer l = net.layers[i];
        if (l.type == CONNECTED && l.batch_normalize) {
            printf("Connected Layer %d\n", i);
            statistics_connected_layer(l);
        }
        if (l.type == GRU && l.batch_normalize) {
            printf("GRU Layer %d\n", i);
            printf("Input Z\n");
            statistics_connected_layer(*l.input_z_layer);
            printf("Input R\n");
            statistics_connected_layer(*l.input_r_layer);
            printf("Input H\n");
            statistics_connected_layer(*l.input_h_layer);
            printf("State Z\n");
            statistics_connected_layer(*l.state_z_layer);
            printf("State R\n");
            statistics_connected_layer(*l.state_r_layer);
            printf("State H\n");
            statistics_connected_layer(*l.state_h_layer);
        }
        printf("\n");
    }
}

void denormalize_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    int i;
    for (i = 0; i < net.n; ++i) {
        layer l = net.layers[i];
        if (l.type == CONVOLUTIONAL && l.batch_normalize) {
            denormalize_convolutional_layer(l);
            net.layers[i].batch_normalize=0;
        }
        if (l.type == CONNECTED && l.batch_normalize) {
            denormalize_connected_layer(l);
            net.layers[i].batch_normalize=0;
        }
        if (l.type == GRU && l.batch_normalize) {
            denormalize_connected_layer(*l.input_z_layer);
            denormalize_connected_layer(*l.input_r_layer);
            denormalize_connected_layer(*l.input_h_layer);
            denormalize_connected_layer(*l.state_z_layer);
            denormalize_connected_layer(*l.state_r_layer);
            denormalize_connected_layer(*l.state_h_layer);
            l.input_z_layer->batch_normalize = 0;
            l.input_r_layer->batch_normalize = 0;
            l.input_h_layer->batch_normalize = 0;
            l.state_z_layer->batch_normalize = 0;
            l.state_r_layer->batch_normalize = 0;
            l.state_h_layer->batch_normalize = 0;
            net.layers[i].batch_normalize=0;
        }
    }
    save_weights(net, outfile);
}

void mkimg(char *cfgfile, char *weightfile, int h, int w, int num, char *prefix)
{
    network net = load_network(cfgfile, weightfile, 0);
    image *ims = get_weights(net.layers[0]);
    int n = net.layers[0].n;
    int z;
    for(z = 0; z < num; ++z){
        image im = make_image(h, w, 3);
        fill_image(im, .5);
        int i;
        for(i = 0; i < 100; ++i){
            image r = copy_image(ims[rand()%n]);
            rotate_image_cw(r, rand()%4);
            random_distort_image(r, 1, 1.5, 1.5);
            int dx = rand()%(w-r.w);
            int dy = rand()%(h-r.h);
            ghost_image(r, im, dx, dy);
            free_image(r);
        }
        char buff[256];
        sprintf(buff, "%s/gen_%d", prefix, z);
        save_image(im, buff);
        free_image(im);
    }
}

void visualize(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    visualize_network(net);
#ifdef OPENCV
    cvWaitKey(0);
#endif
}
/**
    这里是入口
**/
//int main(int argc, char **argv)
int main2(int argc, char **argv)
{
    printf("%s ====%d",argv[0],argc);
    //test_resize("data/bad.png");
    //test_box();
    //test_convolutional_layer();
    if(argc < 2){
        fprintf(stderr, "usage: %s <function>\n", argv[0]);
        return 0;
    }
    gpu_index = find_int_arg(argc, argv, "-i", 0);
    if(find_arg(argc, argv, "-nogpu")) {
        gpu_index = -1;
    }

#ifndef GPU
    gpu_index = -1;
#else
    if(gpu_index >= 0){
        cuda_set_device(gpu_index);
    }
#endif

    if (0 == strcmp(argv[1], "average")){
        average(argc, argv);
    } else if (0 == strcmp(argv[1], "yolo")){
        run_yolo(argc, argv);
    } else if (0 == strcmp(argv[1], "voxel")){
        run_voxel(argc, argv);
    } else if (0 == strcmp(argv[1], "super")){
        run_super(argc, argv);
    } else if (0 == strcmp(argv[1], "lsd")){
        run_lsd(argc, argv);
    } else if (0 == strcmp(argv[1], "detector")){
        //目前用的这个,方法在detector.c里
        run_detector(argc, argv);
    } else if (0 == strcmp(argv[1], "detect")){
        float thresh = find_float_arg(argc, argv, "-thresh", .24);
        char *filename = (argc > 4) ? argv[4]: 0;
        char *outfile = find_char_arg(argc, argv, "-out", 0);
        int fullscreen = find_arg(argc, argv, "-fullscreen");
        test_detector("cfg/coco.data", argv[2], argv[3], filename, thresh, .5, outfile, fullscreen);
    } else if (0 == strcmp(argv[1], "cifar")){
        run_cifar(argc, argv);
    } else if (0 == strcmp(argv[1], "go")){
        run_go(argc, argv);
    } else if (0 == strcmp(argv[1], "rnn")){
        run_char_rnn(argc, argv);
    } else if (0 == strcmp(argv[1], "vid")){
        run_vid_rnn(argc, argv);
    } else if (0 == strcmp(argv[1], "coco")){
        run_coco(argc, argv);
    } else if (0 == strcmp(argv[1], "classify")){
        predict_classifier("cfg/imagenet1k.data", argv[2], argv[3], argv[4], 5);
    } else if (0 == strcmp(argv[1], "classifier")){
        run_classifier(argc, argv);
    } else if (0 == strcmp(argv[1], "regressor")){
        run_regressor(argc, argv);
    } else if (0 == strcmp(argv[1], "art")){
        run_art(argc, argv);
    } else if (0 == strcmp(argv[1], "tag")){
        run_tag(argc, argv);
    } else if (0 == strcmp(argv[1], "compare")){
        run_compare(argc, argv);
    } else if (0 == strcmp(argv[1], "dice")){
        run_dice(argc, argv);
    } else if (0 == strcmp(argv[1], "writing")){
        run_writing(argc, argv);
    } else if (0 == strcmp(argv[1], "3d")){
        composite_3d(argv[2], argv[3], argv[4], (argc > 5) ? atof(argv[5]) : 0);
    } else if (0 == strcmp(argv[1], "test")){
        test_resize(argv[2]);
    } else if (0 == strcmp(argv[1], "captcha")){
        run_captcha(argc, argv);
    } else if (0 == strcmp(argv[1], "nightmare")){
        run_nightmare(argc, argv);
    } else if (0 == strcmp(argv[1], "rgbgr")){
        rgbgr_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "reset")){
        reset_normalize_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "denormalize")){
        denormalize_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "statistics")){
        statistics_net(argv[2], argv[3]);
    } else if (0 == strcmp(argv[1], "normalize")){
        normalize_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "rescale")){
        rescale_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "ops")){
        operations(argv[2]);
    } else if (0 == strcmp(argv[1], "speed")){
        speed(argv[2], (argc > 3 && argv[3]) ? atoi(argv[3]) : 0);
    } else if (0 == strcmp(argv[1], "oneoff")){
        oneoff(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "oneoff2")){
        oneoff2(argv[2], argv[3], argv[4], atoi(argv[5]));
    } else if (0 == strcmp(argv[1], "partial")){
        partial(argv[2], argv[3], argv[4], atoi(argv[5]));
    } else if (0 == strcmp(argv[1], "average")){
        average(argc, argv);
    } else if (0 == strcmp(argv[1], "visualize")){
        visualize(argv[2], (argc > 3) ? argv[3] : 0);
    } else if (0 == strcmp(argv[1], "mkimg")){
        mkimg(argv[2], argv[3], atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), argv[7]);
    } else if (0 == strcmp(argv[1], "imtest")){
        test_resize(argv[2]);
    } else {
        fprintf(stderr, "Not an option: %s\n", argv[1]);
    }
    return 0;
}




#define ISspace(x) isspace((int)(x))

#define SERVER_STRING "Server: jdbhttpd/0.1.0\r\n"

void accept_request(int);
void bad_request(int);
void cat(int, FILE *);
void cannot_execute(int);
void error_die(const char *);
void execute_cgi(int, const char *, const char *, const char *);
int get_line(int, char *, int);
void headers(int, const char *);
void not_found(int);
void serve_file(int, const char *);
int startup(u_short *);
void unimplemented(int);

struct http_header{
    char method[255];
    char *host;
    char uri[255];
    char query_string[1024];
};


struct region_box{
    char *class_name;
    char *xmin;
    char *xmax;
    char *ymin;
    char *ymax;
};




static network static_net;

static char **static_names;
static image **static_alphabet;
static char *static_file_dir="";


void serve_json(int client, const char *json)
{

    char buf[1024];
    //header
    strcpy(buf, "HTTP/1.0 200 OK\r\n");
    send(client, buf, strlen(buf), 0);
    strcpy(buf, SERVER_STRING);
    send(client, buf, strlen(buf), 0);
    sprintf(buf, "Content-Type: text/html\r\n");
    send(client, buf, strlen(buf), 0);
    strcpy(buf, "\r\n");
    send(client, buf, strlen(buf), 0);

    char buf_jon[102400];
    sprintf(buf_jon, "%s\r\n",json);
    send(client, buf_jon, strlen(buf_jon), 0);
    //free(buf);
    //free(json);
    //free(buf_jon);
}

/*
int get_region_boxes_data(image im,int num, float thresh, box *boxes, float **probs, char **names, int classes, char *imageid,char *label_data)
{
    int i;
    int box_num=0;
    for(i = 0; i < num; ++i){
        int class = max_index(probs[i], classes);
        float prob = probs[i][class];
        if(prob > thresh){
            box_num++;
            printf("%s: %.0f%%\n", names[class], prob*100);
            box b = boxes[i];
            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;
            if(left < 0) left = 0;
            if(right > im.w-1) right = im.w-1;
            if(top < 0) top = 0;
            if(bot > im.h-1) bot = im.h-1;
            //printf("%s %d %d %d %d\n", names[class], left,right,top,bot);
            char label_box_data[200];
            sprintf(label_box_data,"%s %d %d %d %d\n", names[class], left,right,top,bot);
            //printf("====label_box_data==\n%s",label_box_data);
            strcat(label_data,label_box_data);
             int width = right-left;
            int height = bot-top;
            if(width<0){
                width=width*-1;
            }
            if(height<0){
                height=height*-1;
            }
            image image_new =make_image(width,height,3);
            cut_box_image(image_new,im,left,right,top,bot);
            char boxs_path[255];
            //FILE *dir = fopen(boxs_path,"r");
            //if(dir==NULL){
            //    mkdir(dir,"rw");
            //}
            sprintf(boxs_path,"%s/InvoiceChardevkit/InvoiceChar2017/JPEGImages/%s_%s",static_file_dir,imageid,names[class]);
            FILE *file = fopen(boxs_path,"r");
            if(file){
                continue;
            }
            printf("========boxs_path:%s\n",boxs_path);
            int new_width=416;
            int new_height=height*new_width/width;
            image_new = resize_image(image_new,new_width,new_height);
            save_image(image_new, boxs_path);
            //free(label_box_data);
            //draw_box_width(im, left, top, right, bot, width, red, green, blue);
        }
    }
    return box_num;
}
*/
struct http_header get_headers(client){
    struct http_header header={};
    int numchars = 1;
    char buf[1024];

    //buf[0] = 'A'; buf[1] = '\0';
    while ((numchars > 0) && strcmp("\n", buf)){  /* read & discard headers */
        numchars = get_line(client, buf, sizeof(buf));
        int length = strlen(buf);
        int i;
        char field[255];
        int j=0;
        char *header_name="";
        for(i=0;i<length;i++){
            int field_length = strlen(field);
            if((':'==buf[i] || ' '==buf[i])&&strcmp(header_name,"")==0){
                j=0;
                if(field_length>0 &&field[i-1]!='\0'){
                    field[i]='\0';
                }
                if(strcmp(field,"GET")==0||strcmp(field,"POST")==0){
                    header_name="method";
                }
            }else{
                field[j]=buf[i];
                j++;
            }
        }
        if(strcmp(header_name,"method")==0){
            printf("=====serve_json=:%s\n",buf);
            int i = 0;
            int j = 0;
            char url[255];
            char method[255];
            char *query_string = NULL;
            while (!ISspace(buf[j]) && (i < sizeof(method) - 1))
            {

                 method[i] = buf[j];
                i++; j++;
            }
            method[i] = '\0';
            strcpy(header.method,method);
            i = 0;
            while (ISspace(buf[j]) && (j < sizeof(buf)))j++;
            while (!ISspace(buf[j]) && (i < sizeof(url) - 1) && (j < sizeof(buf)))
            {
                url[i] = buf[j];
                i++; j++;
            }
            url[i] = '\0';
            query_string = url;
            while ((*query_string != '?') && (*query_string != '\0')){
                query_string++;
            }
            if (*query_string == '?')
            {
                *query_string = '\0';
                query_string++;
            }
            strcpy(header.uri,url);
            strcpy(header.query_string,query_string);
        }
        //printf("=====serve_json=:%s\n",buf);
    }
      //printf("====get_headers===method=%s\n",header.method);
    return header;
}

void accept_request(int client)
{
    try{
        struct http_header header = get_headers(client);
        char *method=header.method;
        char *uri=header.uri;
        time_t timep;
        time (&timep);
        printf("%s===uri:====%s\n",ctime(&timep),uri);
        printf("===method:====%s\n",method);
        char *query_string=header.query_string;
        if (method&&uri&&strcasecmp(method, "GET") == 0&&strcasecmp(uri,"/dk")==0){
            char img_file_path[200];
            strcpy(img_file_path,query_string);
            strcat(img_file_path,".jpg");
            char input[512];
            strcpy(input, img_file_path);
            printf("=====img_file_path==%s\n",img_file_path);
            printf("=====input==%s\n",input);
            char buff[512];
            FILE *resource = fopen(input, "r");
            if (resource == NULL){
                printf("===========not not_found\n");
                not_found(client);
            }else{
                //char out_file_path[200];
                //strcpy(out_file_path,static_file_dir);
                //strcat(out_file_path,"predict");
                char imageid[128];
                int last_index=0;
                int i;
                for(i=0;i<=strlen(query_string);i++){

                    char char_s = query_string[i];

                    if (char_s=='/'){
                        last_index=0;
                    }else{
                        imageid[last_index]=char_s;
                        last_index++;
                    }
                }
                printf("===imageid===:%s\n",imageid);
                float thresh=0.4;
                float hier_thresh=0.0;
                srand(2222222);
                clock_t time;
                int jj;
                float nms=.4;
                printf("====img_file_path===%s\n", img_file_path);
                printf("====input===%s\n", input);
                image im = load_image_color(input,0,0);
                image sized = letterbox_image(im, static_net.w, static_net.h);
                layer l = static_net.layers[static_net.n-1];
                box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
                float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
                for(jj = 0; jj < l.w*l.h*l.n; ++jj) {
                    probs[jj] = calloc(l.classes + 1, sizeof(float *));
                }
                float *X = sized.data;
                time=clock();
                network_predict(static_net, X);
                printf("%s: Predicted in %f seconds.\n", img_file_path, sec(clock()-time));
                get_region_boxes(l, im.w, im.h, static_net.w, static_net.h, thresh, probs, boxes, 0, 0, hier_thresh, 1);
                if (nms) {
                    do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
                }
                char region_boxes_data[1024];
                int boxesnum = get_region_boxes_data(im,l.w*l.h*l.n,thresh,boxes,probs,static_names,l.classes,imageid,region_boxes_data,static_file_dir);

                //draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs, static_names, static_alphabet, l.classes);
                //printf("====region_boxes_data====%s\n",region_boxes_data);
                int r=0;
                int word_index=0;//单个字符的位置
                int word_num_index=0;//一行的第一几个单词
                int word_line_index=0;//第几行
                char *word=calloc(50,sizeof(char));
                struct region_box *region_boxs = calloc(boxesnum,sizeof(struct region_box));
                for(r=0;r<strlen(region_boxes_data);r++){
                    char w = region_boxes_data[r];
                    if(w==' ' || w=='\n'){
                        //char one_word[50];
                        //strcpy(one_word,word);
                        if(word_num_index==0){
                            region_boxs[word_line_index].class_name = word;
                        }
                        if(word_num_index==1){
                            region_boxs[word_line_index].xmin = word;
                        }
                        if(word_num_index==2){
                            region_boxs[word_line_index].xmax = word;
                        }
                        if(word_num_index==3){
                            region_boxs[word_line_index].ymin = word;
                        }
                        if(word_num_index==4){
                            region_boxs[word_line_index].ymax = word;
                        }
                        //printf("========word:%s\n",word);
                        //free(word);
                        word = calloc(50,sizeof(char));
                        word_index=0;
                        word_num_index++;
                        if (w=='\n'){
                            word_num_index=0;
                            word_line_index++;
                        }
                    }else{
                          word[word_index] = w;
                          word_index++;
                    }


                }
                char json[102400];
                strcat(json,"{");
                strcat(json,"   \"success\":0,");
                strcat(json,"   \"result\":[");
                for(i=0;i<boxesnum;i++){
                    char buf[512];
                    sprintf(buf,"{\"class_name\":\"%s\",\"xmin\":%s,\"xmax\":%s,\"ymin\":%s,\"ymax\":%s}",region_boxs[i].class_name,region_boxs[i].xmin,region_boxs[i].xmax,region_boxs[i].ymin,region_boxs[i].ymax);
                    free(region_boxs[i].class_name);
                    free(region_boxs[i].xmin);
                    free(region_boxs[i].xmax);
                    free(region_boxs[i].ymin);
                    free(region_boxs[i].ymax);
                    strcat(json,buf);
                    if(i<boxesnum-1){
                        strcat(json,",");
                    }
                    //free(buf);
                }

                strcat(json,"   ]");
                strcat(json,"}");

                //save_image(im, out_file_path);
                //free(img_file_path);
                //free(out_file_path);
                //free(header);
                //free(region_boxes_data)
                free(region_boxs);

                free_image(im);
                free_image(sized);
                free(boxes);
                free_ptrs((void **)probs, l.w*l.h*l.n);

                fclose(resource);
                resource=NULL;
                serve_json(client,json);
            }



        }else{
            not_found(client);
        }
     }catch{
        bad_request(client);
    }
    close(client);
}


void bad_request(int client)
{
 char buf[1024];

 sprintf(buf, "HTTP/1.0 400 BAD REQUEST\r\n");
 send(client, buf, sizeof(buf), 0);
 sprintf(buf, "Content-type: text/html\r\n");
 send(client, buf, sizeof(buf), 0);
 sprintf(buf, "\r\n");
 send(client, buf, sizeof(buf), 0);
 sprintf(buf, "<P>Your browser sent a bad request, ");
 send(client, buf, sizeof(buf), 0);
 sprintf(buf, "such as a POST without a Content-Length.\r\n");
 send(client, buf, sizeof(buf), 0);

}


void error_die(const char *sc)
{
 perror(sc);
 exit(1);
}


int get_line(int sock, char *buf, int size)
{
 int i = 0;
 char c = '\0';
 int n;

 while ((i < size - 1) && (c != '\n'))
 {
  n = recv(sock, &c, 1, 0);
  /* DEBUG printf("%02X\n", c); */
  if (n > 0)
  {
   if (c == '\r')
   {
    n = recv(sock, &c, 1, MSG_PEEK);
    /* DEBUG printf("%02X\n", c); */
    if ((n > 0) && (c == '\n'))
     recv(sock, &c, 1, 0);
    else
     c = '\n';
   }
   buf[i] = c;
   i++;
  }
  else
   c = '\n';
 }
 buf[i] = '\0';

 return(i);
}


/**********************************************************************/
/* Give a client a 404 not found status message. */
/**********************************************************************/
void not_found(int client)
{
 char buf[1024];

 sprintf(buf, "HTTP/1.0 404 NOT FOUND\r\n");
 send(client, buf, strlen(buf), 0);
 sprintf(buf, SERVER_STRING);
 send(client, buf, strlen(buf), 0);
 sprintf(buf, "Content-Type: text/html\r\n");
 send(client, buf, strlen(buf), 0);
 sprintf(buf, "\r\n");
 send(client, buf, strlen(buf), 0);
 sprintf(buf, "<HTML><TITLE>Not Found</TITLE>\r\n");
 send(client, buf, strlen(buf), 0);
 sprintf(buf, "<BODY><P>The server could not fulfill\r\n");
 send(client, buf, strlen(buf), 0);
 sprintf(buf, "your request because the resource specified\r\n");
 send(client, buf, strlen(buf), 0);
 sprintf(buf, "is unavailable or nonexistent.\r\n");
 send(client, buf, strlen(buf), 0);
 sprintf(buf, "</BODY></HTML>\r\n");
 send(client, buf, strlen(buf), 0);
}



/**********************************************************************/
/* This function starts the process of listening for web connections
 * on a specified port.  If the port is 0, then dynamically allocate a
 * port and modify the original port variable to reflect the actual
 * port.
 * Parameters: pointer to variable containing the port to connect on
 * Returns: the socket */
/**********************************************************************/
int startup(u_short *port)
{
 int httpd = 0;
 struct sockaddr_in name;

 httpd = socket(PF_INET, SOCK_STREAM, 0);
 //BOOL bReuseaddr=TRUE;
 //setsockopt(httpd,SOL_SOCKET ,SO_REUSEADDR,(const char*)&bReuseaddr,sizeof(BOOL));

int opt = 1;
setsockopt( httpd, SOL_SOCKET,SO_REUSEADDR, (const void *)&opt, sizeof(opt) );
 if (httpd == -1)
  error_die("socket");
 memset(&name, 0, sizeof(name));
 name.sin_family = AF_INET;
 name.sin_port = htons(*port);
 name.sin_addr.s_addr = htonl(INADDR_ANY);
 if (bind(httpd, (struct sockaddr *)&name, sizeof(name)) < 0)
  error_die("bind");
 if (*port == 0)  /* if dynamically allocating a port */
 {
  int namelen = sizeof(name);
  if (getsockname(httpd, (struct sockaddr *)&name, &namelen) == -1)
   error_die("getsockname");
  *port = ntohs(name.sin_port);
 }
 if (listen(httpd, 5) < 0)
  error_die("listen");
 return(httpd);
}

/**********************************************************************/
/* Inform the client that the requested web method has not been
 * implemented.
 * Parameter: the client socket */
/**********************************************************************/
void unimplemented(int client)
{
 char buf[1024];

 sprintf(buf, "HTTP/1.0 501 Method Not Implemented\r\n");
 send(client, buf, strlen(buf), 0);
 sprintf(buf, SERVER_STRING);
 send(client, buf, strlen(buf), 0);
 sprintf(buf, "Content-Type: text/html\r\n");
 send(client, buf, strlen(buf), 0);
 sprintf(buf, "\r\n");
 send(client, buf, strlen(buf), 0);
 sprintf(buf, "<HTML><HEAD><TITLE>Method Not Implemented\r\n");
 send(client, buf, strlen(buf), 0);
 sprintf(buf, "</TITLE></HEAD>\r\n");
 send(client, buf, strlen(buf), 0);
 sprintf(buf, "<BODY><P>HTTP request method not supported.\r\n");
 send(client, buf, strlen(buf), 0);
 sprintf(buf, "</BODY></HTML>\r\n");
 send(client, buf, strlen(buf), 0);
}

/**********************************************************************/



int main(int argc, char **argv)
{

    char dir[100];
    int i;
   //把路径保存到字符串s里
   strcpy(dir,argv[0]);
   for(i=strlen(dir); i>0 ; i--){
     if( dir[i] == '/')
       {
            dir[i+1]='\0';
            break;
       }
   }
   static_file_dir=dir;

    if(argc > 1){
        return main2(argc,argv);
    }
    int server_sock = -1;
    u_short port = 8088;
    int client_sock = -1;
    struct sockaddr_in client_name;
    int client_name_len = sizeof(client_name);
    pthread_t newthread;

    server_sock = startup(&port);
    printf("httpd running on port %d\n", port);


    //char *weightfile = "/Users/william/PycharmProjects/darknet_cp/backup/yolo-invoice.backup";
    char weightfile[200];
    char cfgfile[200];
    char datacfg[200];
    strcat(strcpy(weightfile,dir),"backup/yolo-invoice_final.weights");
    strcat(strcpy(cfgfile,dir),"cfg/yolo-invoice-test.cfg");
    strcat(strcpy(datacfg,dir),"cfg/invoice.data");
    static_net = parse_network_cfg(cfgfile);
    load_weights(&static_net, weightfile);
    set_batch_network(&static_net, 1);


    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char names_path[200];
    strcat(strcpy(names_path,dir),name_list);
    static_names = get_labels(names_path);
    static_alphabet = load_alphabet_bydir(static_file_dir);

    while (1){
        client_sock = accept(server_sock,
                           (struct sockaddr *)&client_name,
                           &client_name_len);
        if (client_sock == -1)error_die("accept");
        //accept_request(client_sock);
        if (pthread_create(&newthread, NULL, accept_request, client_sock) != 0)perror("pthread_create");
    }

    close(server_sock);

    return(0);
}

