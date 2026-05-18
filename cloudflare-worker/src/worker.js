export default {
  async fetch(request, env) {
    const incomingUrl = new URL(request.url);
    const upstreamUrl = new URL(env.UPSTREAM_ORIGIN);
    upstreamUrl.pathname = incomingUrl.pathname;
    upstreamUrl.search = incomingUrl.search;

    const headers = new Headers(request.headers);
    headers.delete("host");

    return fetch(
      new Request(upstreamUrl.toString(), {
        method: request.method,
        headers,
        body: request.body,
        redirect: request.redirect,
      }),
    );
  },
};
