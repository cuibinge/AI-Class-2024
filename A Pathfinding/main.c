#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <math.h>
#define ROW 25
#define COL 61
enum derection
{
	up = 1, up_right, right, down_right, down, down_left, left, up_left
};
typedef struct Point
{
	int x, y;
	int Px, Py;
	int F;    //F=G+H
	char list;    //o 开放; c 关闭; n 无状态; b 墙; f 最终道路
}Point;
void F_value_Sub(int x, int y, int x1, int y1, int x2, int y2, Point Map[ROW][COL])
{
	Map[y][x].F = 0;
	int G, H1, H2;
	H1 = (y - y2) * 10; H2 = (x - x2) * 10;
	G = Get_G(x, y, x1, y1, Map);
	Map[y][x].F = G + (int)sqrt(H1 * H1 + H2 * H2);
}
void Sub_region(int x, int y, int x1, int y1, int x2, int y2, Point Map[ROW][COL])
{
	int i1, i2;
	for (i1 = up; i1 <= up_left; i1++)
	{
		switch (i1)
		{
		case up:
			if (y - 1 >= 1 && Map[y - 1][x].list == 'n')
			{
				Map[y - 1][x].list = 'o';
				Map[y - 1][x].Px = x;
				Map[y - 1][x].Py = y;
			}
			else if (y - 1 >= 1 && Map[y - 1][x].list == 'o')
			{
				int Px1 = Map[y - 1][x].Px;
				int Py1 = Map[y - 1][x].Py;
				int F1 = Map[y - 1][x].F;
				Map[y - 1][x].Px = x;
				Map[y - 1][x].Py = y;
				F_value_Sub(x, y - 1, x1, y1, x2, y2, Map);
				if (Map[y - 1][x].F > F1)
				{
					Map[y - 1][x].Px = Px1;
					Map[y - 1][x].Py = Py1;
					Map[y - 1][x].F = F1;
				}
			}
			break;
		case up_right:
			if (y - 1 >= 1 && x + 1 <= COL - 1 && Map[y - 1][x + 1].list == 'n')
			{
				Map[y - 1][x + 1].list = 'o';
				Map[y - 1][x + 1].Px = x;
				Map[y - 1][x + 1].Py = y;
			}
			else if (y - 1 >= 1 && x + 1 <= COL - 1 && Map[y - 1][x + 1].list == 'o')
			{
				int Px1 = Map[y - 1][x + 1].Px;
				int Py1 = Map[y - 1][x + 1].Py;
				int F1 = Map[y - 1][x + 1].F;
				Map[y - 1][x + 1].Px = x;
				Map[y - 1][x + 1].Py = y;
				F_value_Sub(x + 1, y - 1, x1, y1, x2, y2, Map);
				if (Map[y - 1][x + 1].F > F1)
				{
					Map[y - 1][x + 1].Px = Px1;
					Map[y - 1][x + 1].Py = Py1;
					Map[y - 1][x + 1].F = F1;
				}
			}
			break;
		case right:
			if (x + 1 <= COL - 1 && Map[y][x + 1].list == 'n')
			{
				Map[y][x + 1].list = 'o';
				Map[y][x + 1].Px = x;
				Map[y][x + 1].Py = y;
			}
			else if (x + 1 <= COL - 1 && Map[y][x + 1].list == 'o')
			{
				int Px1 = Map[y][x + 1].Px;
				int Py1 = Map[y][x + 1].Py;
				int F1 = Map[y][x + 1].F;
				Map[y][x + 1].Px = x;
				Map[y][x + 1].Py = y;
				F_value_Sub(x + 1, y, x1, y1, x2, y2, Map);
				if (Map[y][x + 1].F > F1)
				{
					Map[y][x + 1].Px = Px1;
					Map[y][x + 1].Py = Py1;
					Map[y][x + 1].F = F1;
				}
			}
			break;
		case down_right:
			if (y + 1 <= ROW - 1 && x + 1 <= COL - 1 && Map[y + 1][x + 1].list == 'n')
			{
				Map[y + 1][x + 1].list = 'o';
				Map[y + 1][x + 1].Px = x;
				Map[y + 1][x + 1].Py = y;
			}
			else if (y + 1 <= ROW - 1 && x + 1 <= COL - 1 && Map[y + 1][x + 1].list == 'o')
			{
				int Px1 = Map[y + 1][x + 1].Px;
				int Py1 = Map[y + 1][x + 1].Py;
				int F1 = Map[y + 1][x + 1].F;
				Map[y + 1][x + 1].Px = x;
				Map[y + 1][x + 1].Py = y;
				F_value_Sub(x + 1, y + 1, x1, y1, x2, y2, Map);
				if (Map[y + 1][x + 1].F > F1)
				{
					Map[y + 1][x + 1].Px = Px1;
					Map[y + 1][x + 1].Py = Py1;
					Map[y + 1][x + 1].F = F1;
				}
			}
			break;
		case down:
			if (y + 1 <= ROW - 1 && Map[y + 1][x].list == 'n')
			{
				Map[y + 1][x].list = 'o';
				Map[y + 1][x].Px = x;
				Map[y + 1][x].Py = y;
			}
			else if (y + 1 <= ROW - 1 && Map[y + 1][x].list == 'o')
			{
				int Px1 = Map[y + 1][x].Px;
				int Py1 = Map[y + 1][x].Py;
				int F1 = Map[y + 1][x].F;
				Map[y + 1][x].Px = x;
				Map[y + 1][x].Py = y;
				F_value_Sub(x, y + 1, x1, y1, x2, y2, Map);
				if (Map[y + 1][x].F > F1)
				{
					Map[y + 1][x].Px = Px1;
					Map[y + 1][x].Py = Py1;
					Map[y + 1][x].F = F1;
				}
			}
			break;
		case down_left:
			if (y + 1 <= ROW - 1 && x - 1 >= 1 && Map[y + 1][x - 1].list == 'n')
			{
				Map[y + 1][x - 1].list = 'o';
				Map[y + 1][x - 1].Px = x;
				Map[y + 1][x - 1].Py = y;
			}
			else if (y + 1 <= ROW - 1 && x - 1 >= 1 && Map[y + 1][x - 1].list == 'o')
			{
				int Px1 = Map[y + 1][x - 1].Px;
				int Py1 = Map[y + 1][x - 1].Py;
				int F1 = Map[y + 1][x - 1].F;
				Map[y + 1][x - 1].Px = x;
				Map[y + 1][x - 1].Py = y;
				F_value_Sub(x - 1, y + 1, x1, y1, x2, y2, Map);
				if (Map[y + 1][x - 1].F > F1)
				{
					Map[y + 1][x - 1].Px = Px1;
					Map[y + 1][x - 1].Py = Py1;
					Map[y + 1][x - 1].F = F1;
				}
			}
			break;
		case left:
			if (x - 1 >= 1 && Map[y][x - 1].list == 'n')
			{
				Map[y][x - 1].list = 'o';
				Map[y][x - 1].Px = x;
				Map[y][x - 1].Py = y;
			}
			else if (x - 1 >= 1 && Map[y][x - 1].list == 'o')
			{
				int Px1 = Map[y][x - 1].Px;
				int Py1 = Map[y][x - 1].Py;
				int F1 = Map[y][x - 1].F;
				Map[y][x - 1].Px = x;
				Map[y][x - 1].Py = y;
				F_value_Sub(x - 1, y, x1, y1, x2, y2, Map);
				if (Map[y][x - 1].F > F1)
				{
					Map[y][x - 1].Px = Px1;
					Map[y][x - 1].Py = Py1;
					Map[y][x - 1].F = F1;
				}
			}
			break;
		case up_left:
			if (y - 1 >= 1 && x - 1 >= 1 && Map[y - 1][x - 1].list == 'n')
			{
				Map[y - 1][x - 1].list = 'o';
				Map[y - 1][x - 1].Px = x;
				Map[y - 1][x - 1].Py = y;
			}
			else if (y - 1 >= 1 && x - 1 >= 1 && Map[y - 1][x - 1].list == 'o')
			{
				int Px1 = Map[y - 1][x - 1].Px;
				int Py1 = Map[y - 1][x - 1].Py;
				int F1 = Map[y - 1][x - 1].F;
				Map[y - 1][x - 1].Px = x;
				Map[y - 1][x - 1].Py = y;
				F_value_Sub(x - 1, y - 1, x1, y1, x2, y2, Map);
				if (Map[y - 1][x - 1].F > F1)
				{
					Map[y - 1][x - 1].Px = Px1;
					Map[y - 1][x - 1].Py = Py1;
					Map[y - 1][x - 1].F = F1;
				}
			}
			break;
		}
	}
}
int Get_G(int x, int y, int x1, int y1, Point Map[ROW][COL])
{
	int G = 0, i, x0 = x, y0 = y;
	for (i = 1; x0 != x1 || y0 != y1; i++)
	{
		if (Map[y0][x0].x != Map[y0][x0].Px && Map[y0][x0].y != Map[y0][x0].Py)
		{
			G += 14;
		}
		else
		{
			G += 10;
		}
		int next_x = Map[y0][x0].Px;
		int next_y = Map[y0][x0].Py;
		x0 = next_x;
		y0 = next_y;
	}
	return G;
}
void F_value(int x1, int y1, int x2, int y2, Point Map[ROW][COL])
{
	int i1, i2;
	for (i1 = 1; i1 <= ROW - 1; i1++)
	{
		for (i2 = 1; i2 <= COL - 1; i2++)
		{
			if (Map[i1][i2].list == 'o' && Map[i1][i2].F == -1)
			{
				Map[i1][i2].F = 0;
				int G, H1, H2;
				H1 = (i1 - y2) * 10; H2 = (i2 - x2) * 10;
				G = Get_G(i2, i1, x1, y1, Map);
				Map[i1][i2].F = G + (int)sqrt(H1 * H1 + H2 * H2);
			}
		}
	}
}
Point C_choose(Point Map[ROW][COL])
{
	int i1, i2, min, flag = 0;
	Point point;
	for (i1 = 1; i1 <= ROW - 1; i1++)
	{
		for (i2 = 1; i2 <= COL - 1; i2++)
		{
			if (flag == 0 && Map[i1][i2].list == 'o')
			{
				min = Map[i1][i2].F;
				flag = 1;
			}
			else if (Map[i1][i2].list == 'o' && Map[i1][i2].F <= min)
			{
				min = Map[i1][i2].F;
			}
		}
	}
	for (i1 = 1; i1 <= ROW - 1; i1++)
	{
		for (i2 = 1; i2 <= COL - 1; i2++)
		{
			if (Map[i1][i2].F == min && Map[i1][i2].list == 'o')
			{
				Map[i1][i2].list = 'c';
				point = Map[i1][i2];
				return point;
			}
		}
	}
}
void Astar_Finding_the_Way(int x1, int y1, int x2, int y2, Point Map[ROW][COL])    //x1,y1 起点; x2,y2 终点
{
	int i1, i2;
	Map[y1][x1].list = 'c';
	Point point = Map[y1][x1];
	int Tx = x1, Ty = y1;
	while (Map[y2][x2].list != 'c')
	{
		Sub_region(Tx, Ty, x1, y1, x2, y2, Map);
		F_value(x1, y1, x2, y2, Map);
		point = C_choose(Map);
		Tx = point.x; Ty = point.y;
	}
}
void Get_final_path(int x1, int y1, int x2, int y2, Point Map[ROW][COL])
{
	int i, x0 = x2, y0 = y2;
	for (i = 1; x0 != x1 || y0 != y1; i++)
	{
		Map[y0][x0].list = 'f';
		int next_x = Map[y0][x0].Px;
		int next_y = Map[y0][x0].Py;
		x0 = next_x;
		y0 = next_y;
	}
	Map[y1][x1].list = 'f';
}
void Generate_map(Point Map[ROW][COL])
{
	int i1, i2;
	for (i1 = 1; i1 <= ROW - 1; i1++)
	{
		for (i2 = 1; i2 <= COL - 1; i2++)
		{
			Map[i1][i2].x = i2; Map[i1][i2].y = i1; Map[i1][i2].F = -1;
			if (
				(i2 == 3 && i1 <= 6) ||
				(i2 == 3 && i1 >= 18) ||
				(i2 == 6 && i1 <= 6) ||
				(i2 == 6 && i1 >= 9 && i1 <= 21) ||
				(i2 == 9 && i1 >= 12 && i1 <= 15) ||
				(i2 == 12 && i1 <= 9) ||
				(i2 == 21 && i1 <= 3) ||
				(i2 == 21 && i1 >= 18) ||
				(i2 == 30 && i1 <= 15) ||
				(i2 == 30 && i1 >= 18) ||
				(i2 == 45 && i1 >= 18) ||
				(i2 == 54 && i1 >= 12) ||
				(i1 == 6 && i2 >= 30 && i2 <= 45) ||
				(i1 == 6 && i2 >= 9 && i2 <= 21) ||
				(i1 == 9 && i2 >= 30 && i2 <= 45) ||
				(i1 == 9 && i2 >= 18 && i2 <= 42) ||
				(i1 == 9 && i2 >= 51) ||
				(i1 == 15 && i2 >= 24 && i2 <= 36) ||
				(i1 == 21 && i2 >= 27 && i2 <= 42) ||
				(i1 == 24 && i2 >= 30 && i2 <= 42) ||
				(i1 == 24 && i2 >= 9 && i2 <= 21)
				)
			{
				Map[i1][i2].list = 'b';
			}
			else
				Map[i1][i2].list = 'n';
		}
	}
}
void PrintMap(Point Map[ROW][COL])
{
	int i1, i2;
	printf(" ");
	for (i1 = 1; i1 <= COL - 1; i1++) { printf("-"); }
	printf(" \n");
	for (i1 = 1; i1 <= ROW - 1; i1++)
	{
		printf("|");
		for (i2 = 1; i2 <= COL - 1; i2++)
		{
			if (Map[i1][i2].list == 'b')
				printf("X");
			else if (Map[i1][i2].list == 'f')
				printf("*");
			else
				printf(" ");
		}
		printf("|\n");
	}
	printf(" ");
	for (i1 = 1; i1 <= COL - 1; i1++) { printf("-"); }
	printf(" \n");
}
void Input_start_end_point(int* x1, int* y1, int* x2, int* y2)
{
	printf("起点: ");
	scanf("%d %d", x1, y1);
	printf("终点: ");
	scanf("%d %d", x2, y2);
}
int main()
{
	Point Map[ROW][COL];
	Generate_map(Map);
	int startpoint_x, startpoint_y, destination_x, destination_y;
	printf("当前地图: \n");
	PrintMap(Map);
	Input_start_end_point(&startpoint_x, &startpoint_y, &destination_x, &destination_y);
	Astar_Finding_the_Way(startpoint_x, startpoint_y, destination_x, destination_y, Map);
	Get_final_path(startpoint_x, startpoint_y, destination_x, destination_y, Map);
	PrintMap(Map);
}