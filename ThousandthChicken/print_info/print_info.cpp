#include <stdio.h>
#include <stdarg.h>
#include <time.h>

//struct timeval start_time;

void println(const char* file, const char* function, const int line, const char *str)
{
	time_t timer;
    time(&timer);  /* get current time; same as: timer = time(NULL)  */
	fprintf(stdout, "[%s] (%s:%d) %s\n", function, file, line, str);
}

void println_var(const char* file, const char* function, const int line, const char* format, ...)
{
	char status[512];
	va_list arglist;

	va_start(arglist, format);
	vsprintf(status, format, arglist);
	va_end(arglist);

	println(file, function, line, status);
}

void println_start(const char* file, const char* function, const int line)
{
	println(file, function, line, "start");
}

void println_end(const char* file, const char* function, const int line)
{
	println(file, function, line, "end");
}

/*void start_measure()
{
	gettimeofday(&start_time, NULL);
}*/

long int start_measure()
{
/*	struct timeval start;
	gettimeofday(&start, NULL);

	return start.tv_sec * 1000000 + start.tv_usec;*/
	return 0;
}

long int stop_measure(long int start)
{
//	struct timeval end;
//	gettimeofday(&end, NULL);
//	long int time = (end.tv_sec * 1000000 + end.tv_usec) - start;
//
//	return time;
	return 0;
}

/*long int stop_measure(const char* file, const char* function, const int line)
{
	struct timeval end_time;
	gettimeofday(&end_time, NULL);
	long int time = (end_time.tv_sec - start_time.tv_sec) * 1000000 + end_time.tv_usec - start_time.tv_usec;
	println_var(file, function, line, "Computation time:%ld", time);

	return time;
}*/

long int stop_measure_msg(const char* file, const char* function, const int line, char *msg)
{
	/*struct timeval end_time;
	gettimeofday(&end_time, NULL);
	long int time = (end_time.tv_sec - start_time.tv_sec) * 1000000 + end_time.tv_usec - start_time.tv_usec;
	println_var(file, function, line, "%s:%ld", msg, time);

	return time;*/
	return 0;
}

long int stop_measure_no_info()
{
	//struct timeval end_time;
	/*gettimeofday(&end_time, NULL);
	long int time = (end_time.tv_sec - start_time.tv_sec) * 1000000 + end_time.tv_usec - start_time.tv_usec;
	printf("%ld\n", time);

	return time;*/
	return 0;
}
