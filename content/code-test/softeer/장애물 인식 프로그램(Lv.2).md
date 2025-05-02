+++
date = '2025-05-01T14:05:19+09:00'
draft = false
title = '[softeer] 장애물 인식 프로그램(Lv.2) - Java'
+++

## 문제 설명
|  |  |  |  |  |  |  |
|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 0 | 1 | 1 | 1 |
| 0 | 1 | 1 | 0 | 1 | 0 | 1 |
| 0 | 1 | 1 | 0 | 1 | 0 | 1 |
| 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| 0 | 1 | 1 | 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | 1 | 1 | 1 | 0 |
| 0 | 1 | 1 | 0 | 0 | 0 | 0 |
> 우선 [그림 1]과 같이 정사각형 모양의 지도가 있다. 1은 장애물이 있는 곳을, 0은 도로가 있는 곳을 나타낸다.
당신은 이 지도를 가지고 연결된 장애물들의 모임인 블록을 정의하고, 불록에 번호를 붙이려 한다. 여기서 연결되었다는 것은 어떤 장애물이 좌우, 혹은 아래위로 붙어 있는 경우를 말한다. 대각선 상에 장애물이 있는 경우는 연결된 것이 아니다.
**첫 번째 줄에는 총 블록 수를 출력** 하시오.
그리고 **각 블록 내 장애물의 수를 오름차순으로 정렬하여 한 줄에 하나씩 출력**하시오.


## 문제 풀이
- 지도로 제공된 곳에서 뭉텅이? 덩어리?를 경계선으로 부터 구분해야함. -> BFS로 풀이해보자....
- 총 뭉텅이의 수를 알아야하고, 그 뭉텅이별의 너비를 알고있어야함. -> 탐색 경로에 대해서 저장해보자...

```java
public class Main {
    public static void main(String[] args) throws Exception {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        StringTokenizer st = new StringTokenizer(br.readLine());
        int length = Integer.parseInt(st.nextToken());
        int[][] maps = new int[length][length];
        boolean[][] visited = new boolean[length][length];

        // 지도 입력 받기, 방문 히스토리도,,,
        for (int i = 0; i < length; i++) {
            String[] splitedLines = br.readLine().split("");
            for (int k = 0; k < length; k++) {
                int value = Integer.parseInt(splitedLines[k]);
                if (value != 0) {
                    maps[i][k] = value;
                    continue;
                }
                visited[i][k] = true;
            }
        }
        List<Integer> res = new ArrayList();
        for (int i = 0; i < length; i++) {
            for (int k = 0; k < length; k++) {
                // 들렸던 곳이거나,,,, 가면 안되는곳이거나,,
                if (visited[i][k] || maps[i][k] == 0) {
                    visited[i][k] = true;
                    continue;
                }
                visited[i][k] = true;
                Queue<int[]> queue = new LinkedList();
                queue.offer(new int[]{k, i});
                int maxDistance = 1;
                // x,y 방향
                int[][] futureArr = new int[][]{{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
                while (!queue.isEmpty()) {
                    int[] elem = queue.poll();
                    for (int[] future : futureArr) {
                        int x = elem[0] + future[0];
                        int y = elem[1] + future[1];
                        // 앞으로 가야할 곳이 지도 경계 밖이거나....
                        if ((x < 0 || x >= length) || (y < 0 || y >= length) || maps[y][x] == 0) {
                            continue;
                        }
                        // 이제 갈거니까 방문 처리
                        if (!visited[y][x]) {
                            visited[y][x] = true;
                            maxDistance++;
                            queue.offer(new int[]{x, y});
                        }
                    }
                }
                // 방문한 뭉텅이의 너비를 저장하자.
                res.add(maxDistance);
            }
        }
        System.out.println(res.size());
        res = res.stream().sorted().collect(Collectors.toList());
        for (Integer elem : res) {
            System.out.println(elem);
        }
    }
}
```

### 수행 결과

| NO | 이름  | 결과 | 실행시간  | 메모리   |
|----|-------|------|-----------|----------|
| 1  | TC1   | 정답 | 0.081 초  | 10.86 MB |
| 2  | TC10  | 정답 | 0.080 초  | 11.27 MB |
| 3  | TC2   | 정답 | 0.072 초  | 10.96 MB |
| 4  | TC3   | 정답 | 0.098 초  | 11.71 MB |
| 5  | TC4   | 정답 | 0.080 초  | 11.36 MB |
| 6  | TC5   | 정답 | 0.072 초  | 10.96 MB |
| 7  | TC6   | 정답 | 0.085 초  | 11.36 MB |
| 8  | TC7   | 정답 | 0.077 초  | 11.08 MB |
| 9  | TC8   | 정답 | 0.075 초  | 11.08 MB |
| 10 | TC9   | 정답 | 0.081 초  | 11.34 MB |


